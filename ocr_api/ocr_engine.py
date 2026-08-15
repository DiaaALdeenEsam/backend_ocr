import os
import gc

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights')
BASE_MODEL_NAME = "sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3"
MAX_DIMENSION = 1280
def active_train_on_examples(
    examples,
    base_model_dir,
    adapter_dir,
    output_root=None,
    num_epochs=2,
    batch_size=1,
    learning_rate=1e-5,
    device=None,
    val_fraction=0.1,
    allow_promotion_without_val=False,
    use_amp=False,
    grad_accum_steps=1,
):
    """
    Runs a short QLoRA/LoRA fine-tuning pass on collected correction examples
    and saves the updated adapter to a new timestamped directory. Evaluation on
    a validation split is used to decide whether to promote the new adapter.
    Returns dict with keys: success, new_weights_dir, baseline_cer, new_cer, promoted
    """
    try:
        import os
        import torch
        from datetime import datetime
        from torch.utils.data import Dataset, DataLoader
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel
    except Exception as e:
        return {'success': False, 'new_weights_dir': None, 'error': f'import failed: {e}'}

    if not examples:
        return {'success': False, 'new_weights_dir': None, 'error': 'no examples'}

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if output_root is None:
        output_root = os.path.join(BASE_DIR, 'weights_active_runs')
    os.makedirs(output_root, exist_ok=True)

    def levenshtein(a, b):
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        prev = list(range(lb + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i] + [0] * lb
            for j, cb in enumerate(b, start=1):
                add = prev[j] + 1
                delete = cur[j-1] + 1
                change = prev[j-1] + (0 if ca == cb else 1)
                cur[j] = min(add, delete, change)
            prev = cur
        return prev[lb]

    def cer(preds, refs):
        total_edits = 0
        total_chars = 0
        for p, r in zip(preds, refs):
            total_edits += levenshtein(p, r)
            total_chars += max(1, len(r))
        return float(total_edits) / float(total_chars)

    class CorrectionDataset(Dataset):
        def __init__(self, examples, processor, instruction="Text Recognition:"):
            self.examples = examples
            self.processor = processor
            self.instruction = instruction

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]
            image = Image.open(ex['image_path']).convert('RGB')

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.instruction},
                ]},
                {"role": "assistant", "content": ex['edited_text']},
            ]

            full_ids = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt",
            )

            prompt_only = self.processor.apply_chat_template(
                messages[:1], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            )
            prompt_len = prompt_only["input_ids"].shape[1]

            input_ids = full_ids["input_ids"].squeeze(0)
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            item = {
                "input_ids": input_ids,
                "attention_mask": full_ids["attention_mask"].squeeze(0),
                "labels": labels,
            }
            if "pixel_values" in full_ids:
                item["pixel_values"] = full_ids["pixel_values"]
                item["image_grid_thw"] = full_ids["image_grid_thw"]
            return item

    def collate_fn(batch, pad_token_id):
        max_len = max(b["input_ids"].shape[0] for b in batch)

        def pad(t, value):
            pad_len = max_len - t.shape[0]
            return torch.cat([t, torch.full((pad_len,), value, dtype=t.dtype)]) if pad_len > 0 else t

        input_ids = torch.stack([pad(b["input_ids"], pad_token_id) for b in batch])
        attention_mask = torch.stack([pad(b["attention_mask"], 0) for b in batch])
        labels = torch.stack([pad(b["labels"], -100) for b in batch])

        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
            out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
        return out

    # ensure image_path exists for all examples
    for ex in examples:
        if not ex.get('image_path') or not os.path.exists(ex.get('image_path')):
            return {'success': False, 'new_weights_dir': None, 'error': f"missing image_path for example {ex.get('ocr_record_id')}"}

    # split train/val
    import random
    rng = random.Random(42)
    ex_copy = examples.copy()
    rng.shuffle(ex_copy)
    val_size = max(1, int(len(ex_copy) * val_fraction)) if len(ex_copy) > 1 else 0
    val_examples = ex_copy[:val_size]
    train_examples = ex_copy[val_size:]
    if not train_examples:
        train_examples = val_examples
        val_examples = []

    try:
        # device_map handling
        if isinstance(device, str) and device == 'cuda' and torch.cuda.is_available():
            device_map = {"": 0}
            torch_dtype = torch.bfloat16
        else:
            device_map = 'cpu'
            torch_dtype = torch.float32

        processor = AutoProcessor.from_pretrained(base_model_dir)

        # baseline model (with current adapter if present)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_dir, torch_dtype=torch_dtype, device_map=device_map,
        )
        if adapter_dir and os.path.exists(adapter_dir):
            baseline_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
        else:
            baseline_model = base_model
        baseline_model.to(device)
        baseline_model.eval()

        # evaluate baseline on val
        baseline_preds = []
        baseline_refs = []
        if val_examples:
            for ex in val_examples:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": Image.open(ex['image_path']).convert('RGB')},
                        {"type": "text", "text": "Text Recognition:"},
                    ]}
                ]
                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        generated_ids = baseline_model.generate(**inputs, max_new_tokens=256)
                    # attempt to trim prompt tokens if present
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.get('input_ids', []), generated_ids)] if inputs.get('input_ids', None) is not None else generated_ids
                    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    baseline_preds.append(output_text[0].strip())
                    baseline_refs.append(ex['edited_text'])
                except Exception:
                    baseline_preds.append('')
                    baseline_refs.append(ex['edited_text'])
            baseline_cer = cer(baseline_preds, baseline_refs)
        else:
            baseline_cer = float('inf')

        # prepare model for training
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_dir, torch_dtype=torch_dtype, device_map=device_map,
        )
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        model.to(device)
        model.train()

        dataset = CorrectionDataset(train_examples, processor)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, processor.tokenizer.pad_token_id),
        )

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

        scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(**batch)
                        loss = outputs.loss / grad_accum_steps
                    scaler.scale(loss).backward()
                    if (step + 1) % grad_accum_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = model(**batch)
                    loss = outputs.loss / grad_accum_steps
                    loss.backward()
                    if (step + 1) % grad_accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                epoch_loss += loss.item() * (grad_accum_steps if scaler is None else 1)

            avg_loss = epoch_loss / max(1, len(loader))
            print(f"[active_train] epoch {epoch+1}/{num_epochs} avg_loss={avg_loss:.4f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_weights_dir = os.path.join(output_root, f"adapter_{timestamp}")
        os.makedirs(new_weights_dir, exist_ok=True)
        model.save_pretrained(new_weights_dir)

        # evaluate new model on val
        if val_examples:
            new_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_dir, torch_dtype=torch_dtype, device_map=device_map)
            new_model = PeftModel.from_pretrained(new_base, new_weights_dir, is_trainable=False)
            new_model.to(device)
            new_model.eval()

            new_preds = []
            new_refs = []
            for ex in val_examples:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": Image.open(ex['image_path']).convert('RGB')},
                        {"type": "text", "text": "Text Recognition:"},
                    ]}
                ]
                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        generated_ids = new_model.generate(**inputs, max_new_tokens=256)
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.get('input_ids', []), generated_ids)] if inputs.get('input_ids', None) is not None else generated_ids
                    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    new_preds.append(output_text[0].strip())
                    new_refs.append(ex['edited_text'])
                except Exception:
                    new_preds.append('')
                    new_refs.append(ex['edited_text'])

            new_cer = cer(new_preds, new_refs)
        else:
            new_cer = float('inf')

        if not val_examples and not allow_promotion_without_val:
            promoted = False
        else:
            promoted = (new_cer < baseline_cer)

        return {
            'success': bool(promoted),
            'new_weights_dir': new_weights_dir if promoted else None,
            'baseline_cer': baseline_cer,
            'new_cer': new_cer,
            'promoted': promoted,
        }

    except Exception as e:
        return {'success': False, 'new_weights_dir': None, 'error': str(e)}
    

# lightweight inference engine factory
_OCR_ENGINE = None
_OCR_ENGINE_LOCK = None

def get_ocr_engine(device=None):
    """Return a singleton OCR engine with a simple `predict(image_path)` method.

    This is intentionally lightweight and lazy: it loads the processor and model
    on first use and caches them. It will use the adapter in `WEIGHTS_PATH` if present.
    """
    global _OCR_ENGINE, _OCR_ENGINE_LOCK
    import threading
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    if _OCR_ENGINE_LOCK is None:
        _OCR_ENGINE_LOCK = threading.Lock()

    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            return _OCR_ENGINE

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
            if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
                model = PeftModel.from_pretrained(base_model, WEIGHTS_PATH, is_trainable=False)
            else:
                model = base_model
            model.to(device)
            model.eval()
        except Exception as e:
            raise

        class _Engine:
            def __init__(self, model, processor, device):
                self.model = model
                self.processor = processor
                self.device = device

            def predict(self, image_path):
                img = Image.open(image_path).convert('RGB')
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Text Recognition:"},
                ]}]

                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=256)

                # trim prompt tokens if present
                if inputs.get('input_ids', None) is not None:
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.get('input_ids', []), generated_ids)]
                else:
                    generated_ids_trimmed = generated_ids

                output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return output_text[0].strip() if output_text else ''

        _OCR_ENGINE = _Engine(model, processor, device)
        return _OCR_ENGINE


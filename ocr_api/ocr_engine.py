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

class OCREngine:
    _instance = None

    def __init__(self):
        print("⚡ Loading Base Qwen2.5-VL Model on GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. إعدادات BitsAndBytes الخاصة بـ 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            llm_int8_enable_fp32_cpu_offload=True  # تمكين الـ CPU Offload لتفادي خطأ الـ Offload
        )
        
        # 2. تحديد خريطة الأجهزة: إجبار التسكين على GPU 0 لتفادي التوزيع العشوائي
        device_map = {"": 0} if self.device == "cuda" else "cpu"

        # 3. تحميل النموذج الأساسي
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config if self.device == "cuda" else None,
            device_map=device_map,
            low_cpu_mem_usage=True,
            tie_word_embeddings=True
        )
        
        # 4. تحميل أوزان LoRA Adapter وتطبيقها
        print("⚡ Applying Fine-Tuned LoRA Weights from 'weights/'...")
        if os.path.exists(WEIGHTS_PATH):
            self.model = PeftModel.from_pretrained(
                base_model, 
                WEIGHTS_PATH,
                device_map=device_map,
                is_trainable=False
            )
        else:
            self.model = base_model

        self.processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
        print(f"✅ OCR Engine Ready on device: {self.device}")

    @staticmethod
    def _resize_image_for_ocr(image_source):
        if isinstance(image_source, Image.Image):
            image = image_source.copy()
            original_size = image.size
        else:
            image = Image.open(image_source)
            original_size = image.size

        width, height = image.size
        if max(width, height) <= MAX_DIMENSION:
            return image, original_size

        resized = image.copy()
        resized.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        print(f"Original image size: {width}x{height} -> resized to {resized.size[0]}x{resized.size[1]}")
        return resized, original_size

    @torch.inference_mode()
    def predict(self, image_path: str) -> str:
        image_source = image_path
        pil_image, original_size = self._resize_image_for_ocr(image_source)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "اقرأ النص العربي المكتوب بخط اليد في هذه الصورة واكتبه بدقة."},
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            return output_text[0].strip()
        finally:
            if isinstance(image_path, str):
                if hasattr(pil_image, 'close'):
                    pil_image.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

def get_ocr_engine():
    if OCREngine._instance is None:
        OCREngine._instance = OCREngine()
    return OCREngine._instance
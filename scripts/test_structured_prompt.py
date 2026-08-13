import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ocr_api.ocr_engine import get_ocr_engine

MAX_DIMENSION = 1280

IMAGE_DIRS = [
    ROOT / 'media' / 'ocr_images' / '2026' / '08' / '13',
    ROOT / 'media' / 'ocr_images' / '2026' / '08' / '03',
    ROOT / 'media' / 'ocr_images' / '2026' / '08' / '02',
]

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

PROMPTS = {
    'A': (
        'اقرأ النص العربي المكتوب بخط اليد في هذه الصورة، واكتبه بدقة، مع تقسيم النص إلى فقرات منفصلة. '
        'أرجع النتيجة بصيغة Markdown، بحيث تفصل كل فقرة بسطر فارغ.'
    ),
    'B': (
        'اقرأ النص العربي المكتوب بخط اليد في هذه الصورة بدقة، وأرجع النتيجة فقط بصيغة JSON بدون أي شرح إضافي، '
        'بالشكل التالي: {"paragraphs": ["نص الفقرة الأولى", "نص الفقرة الثانية"]}'
    ),
}


def find_image_files():
    images = []
    for directory in IMAGE_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            name_lower = path.name.lower()
            if name_lower == 'debug_input.jpg' or name_lower.startswith('debug_input'):
                continue
            images.append(path)

    return images


def resolve_image_path(image_arg):
    candidates = []

    if image_arg:
        candidates.append(Path(image_arg))
        if not Path(image_arg).is_absolute():
            candidates.append(ROOT / image_arg)
            candidates.append(ROOT / 'media' / 'ocr_images' / image_arg)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    images = find_image_files()
    if not images:
        raise FileNotFoundError(
            'No non-debug sample images were found under media/ocr_images/2026/08/13/, '
            'media/ocr_images/2026/08/03/, or media/ocr_images/2026/08/02/.'
        )

    print('Available images (excluding debug_input.jpg):')
    for img in images:
        rel = img.relative_to(ROOT)
        print(f' - {rel.as_posix()}')
    raise FileNotFoundError('Please rerun with --image pointing to one of the files above.')


def resize_image_for_ocr(image_path, max_dimension=MAX_DIMENSION):
    source = Image.open(image_path)
    try:
        original_width, original_height = source.size
        if max(original_width, original_height) <= max_dimension:
            return image_path, (original_width, original_height)

        resized = source.copy()
        resized.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
        resized_path = image_path.with_name(f'{image_path.stem}_resized{image_path.suffix}')
        resized.save(resized_path, format='JPEG' if image_path.suffix.lower() in {'.jpg', '.jpeg'} else None)
        return resized_path, (original_width, original_height)
    finally:
        source.close()


def run_prompt_with_custom_text(engine, image_path, prompt_text):
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': str(image_path)},
                {'type': 'text', 'text': prompt_text},
            ],
        }
    ]

    text = engine.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = __import__('qwen_vl_utils').process_vision_info(messages)
    inputs = engine.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt'
    ).to(engine.device)

    generated_ids = engine.model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = engine.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    del inputs
    del generated_ids
    del generated_ids_trimmed
    del image_inputs
    del video_inputs
    return output_text[0].strip()


def run_prompt(engine, image_path, label, prompt_text):
    print(f'\n========================================')
    print(f'IMAGE: {image_path}')
    print(f'PROMPT {label}:')
    print(prompt_text)
    print('RAW OUTPUT START')
    try:
        result = run_prompt_with_custom_text(engine, image_path, prompt_text)
        print(result)
    except Exception as exc:
        print(f'ERROR: {exc}')
    print('RAW OUTPUT END')
    print('========================================\n')


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Run OCR prompt tests on a single image.')
    parser.add_argument('--image', dest='image', help='Image path relative to media/ocr_images or full path to a file.')
    args = parser.parse_args()

    try:
        image_path = resolve_image_path(args.image)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    image_path_for_run, original_size = resize_image_for_ocr(image_path)
    if image_path_for_run != image_path:
        print(f'Resized from {original_size[0]}x{original_size[1]} to {Image.open(image_path_for_run).size[0]}x{Image.open(image_path_for_run).size[1]}')
        Image.open(image_path_for_run).close()

    engine = get_ocr_engine()
    try:
        for label, prompt_text in PROMPTS.items():
            run_prompt(engine, image_path_for_run, label, prompt_text)
    finally:
        cleanup_cuda()
        print(f'GPU cache cleared after processing: {image_path_for_run}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

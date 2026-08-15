from django.core.management.base import BaseCommand
from django.utils import timezone

from ocr_api.models import EditedOCRExample
from ocr_api.tasks import run_active_training


class Command(BaseCommand):
    help = 'Run active training loop on collected user edits.'

    def add_arguments(self, parser):
        parser.add_argument('--threshold', type=int, default=10, help='Minimum number of edits to trigger training')

    def handle(self, *args, **options):
        threshold = options.get('threshold', 10)
        pending_qs = EditedOCRExample.objects.filter(used=False).order_by('created_at')
        total_pending = pending_qs.count()

        if total_pending < threshold:
            self.stdout.write(self.style.NOTICE(f'Not enough pending edits ({total_pending}) to trigger training (threshold={threshold}).'))
            return

        examples = []
        for edit in pending_qs:
            record = edit.ocr_record
            image_path = None
            try:
                image_path = record.image.path if record.image and hasattr(record.image, 'path') else None
            except Exception:
                image_path = None

            examples.append({
                'ocr_record_id': record.id,
                'original_text': record.extracted_text or '',
                'edited_text': edit.edited_text or '',
                'image_path': image_path,
                'user_id': edit.user_id,
            })

        # enqueue the Celery task to process training asynchronously
        run_active_training.delay(threshold)
        self.stdout.write(self.style.SUCCESS('Enqueued active training task.'))

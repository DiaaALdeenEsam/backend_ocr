from django.db import models

class OCRRecord(models.Model):
    STATUS_UPLOADING = 'uploading'
    STATUS_LOADING_MODEL = 'loading_model'
    STATUS_PROCESSING = 'processing'
    STATUS_SEGMENTING_LINES = 'segmenting_lines'
    STATUS_PROCESSING_LINES = 'processing_lines'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'

    STATUS_CHOICES = [
        (STATUS_UPLOADING, 'Uploading'),
        (STATUS_LOADING_MODEL, 'Loading Model'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_SEGMENTING_LINES, 'Segmenting image into lines'),
        (STATUS_PROCESSING_LINES, 'Recognizing lines'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]

    # حفظ الصورة في مجلد media/ocr_images/YYYY/MM/DD/
    image = models.ImageField(upload_to='ocr_images/%Y/%m/%d/')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_UPLOADING)
    extracted_text = models.TextField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"OCR Record #{self.id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
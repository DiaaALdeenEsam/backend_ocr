import logging
import os

from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.db import models, transaction
from django.db.models import F
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

logger = logging.getLogger(__name__)


class UserProfile(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile',
    )
    total_storage_used = models.BigIntegerField(default=0)
    generated_storage_used = models.BigIntegerField(default=0)

    def __str__(self):
        return f'{self.user.username} — {self.total_storage_used} bytes'


class OCRRecord(models.Model):
    STATUS_PENDING = 'PENDING'
    STATUS_PROCESSING = 'PROCESSING'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_FAILED = 'FAILED'

    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='ocr_records',
    )
    image = models.ImageField(upload_to='ocr_images/')
    file_name = models.CharField(max_length=255, blank=True)
    file_size = models.BigIntegerField(default=0)
    extracted_text = models.TextField(blank=True, null=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
    )
    error_message = models.TextField(blank=True, null=True)
    document_type = models.CharField(max_length=50, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'OCR Record #{self.id} — {self.file_name or "unnamed"}'

    def save(self, *args, **kwargs):
        if self.image:
            if hasattr(self.image, 'size') and self.image.size:
                self.file_size = self.image.size
            if self.image.name:
                self.file_name = os.path.basename(self.image.name)
        super().save(*args, **kwargs)


class GeneratedFile(models.Model):
    """Persisted OCR export (PDF, JSON, Word, Excel).

    Relationship: ``OCRRecord`` (source image) 1 → N ``GeneratedFile``.
    At most one file exists per ``(source_image, file_type)``; re-export overwrites.
    """

    class FileType(models.TextChoices):
        PDF = 'PDF', 'PDF'
        JSON = 'JSON', 'JSON'
        WORD = 'WORD', 'Word'
        EXCEL = 'EXCEL', 'Excel'

    class StorageBackend(models.TextChoices):
        LOCAL = 'local', 'Local'
        S3 = 's3', 'S3'

    MIME_MAP = {
        FileType.PDF: 'application/pdf',
        FileType.JSON: 'application/json',
        FileType.WORD: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        FileType.EXCEL: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    }

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='generated_files',
    )
    source_image = models.ForeignKey(
        OCRRecord,
        on_delete=models.CASCADE,
        related_name='generated_files',
    )
    file_name = models.CharField(max_length=255)
    file = models.FileField(upload_to='generated_files/%Y/%m/', blank=True, max_length=1024)
    file_type = models.CharField(max_length=16, choices=FileType.choices)
    mime_type = models.CharField(max_length=128)
    file_size_bytes = models.BigIntegerField(default=0)
    storage_backend = models.CharField(
        max_length=16,
        choices=StorageBackend.choices,
        default=StorageBackend.LOCAL,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'generated_files'
        ordering = ['-created_at']
        constraints = [
            models.UniqueConstraint(
                fields=['source_image', 'file_type'],
                name='uq_generated_file_source_type',
            ),
            models.CheckConstraint(
                condition=models.Q(file_type__in=['PDF', 'JSON', 'WORD', 'EXCEL']),
                name='ck_generated_file_type',
            ),
            models.CheckConstraint(
                condition=models.Q(file_size_bytes__gte=0),
                name='ck_generated_file_size_non_negative',
            ),
        ]
        indexes = [
            models.Index(fields=['user', '-created_at'], name='idx_gf_user_created'),
            models.Index(fields=['user', 'file_type'], name='idx_gf_user_type'),
            models.Index(fields=['source_image'], name='idx_gf_source'),
            models.Index(
                fields=['user', 'file_type', '-created_at'],
                name='idx_gf_user_type_created',
            ),
        ]

    def __str__(self):
        return f'{self.file_type} #{self.id} — {self.file_name}'

    def save(self, *args, **kwargs):
        self._previous_file_size_bytes = 0
        if self.pk:
            previous = (
                type(self).objects.filter(pk=self.pk)
                .values_list('file_size_bytes', flat=True)
                .first()
            )
            if previous is not None:
                self._previous_file_size_bytes = previous
        if not self.mime_type:
            self.mime_type = self.MIME_MAP.get(self.file_type, 'application/octet-stream')
        super().save(*args, **kwargs)

    @classmethod
    def upsert_from_bytes(cls, ocr_record, file_type, content, file_name):
        """Persist or replace a generated export using ``update_or_create``.

        Unique on ``(source_image, file_type)``: regenerating the same format
        replaces the previous blob and updates ``file_size_bytes``.
        """
        if isinstance(content, memoryview):
            content = content.tobytes()
        elif not isinstance(content, bytes):
            content = bytes(content)

        mime_type = cls.MIME_MAP.get(file_type, 'application/octet-stream')

        with transaction.atomic():
            instance, _created = cls.objects.update_or_create(
                source_image=ocr_record,
                file_type=file_type,
                defaults={
                    'user': ocr_record.user,
                    'file_name': file_name,
                    'mime_type': mime_type,
                    'storage_backend': cls.StorageBackend.LOCAL,
                },
            )
            if instance.file:
                instance.file.delete(save=False)

            instance.file.save(file_name, ContentFile(content), save=False)
            instance.file_size_bytes = instance.file.size or len(content)
            instance.file_name = file_name
            instance.mime_type = mime_type
            instance.save()
            return instance


def persist_generated_export(ocr_record, file_type, content, file_name):
    """Best-effort persist used by export/download views. Never raises."""
    try:
        return GeneratedFile.upsert_from_bytes(
            ocr_record=ocr_record,
            file_type=file_type,
            content=content,
            file_name=file_name,
        )
    except Exception:
        logger.exception(
            'Failed to persist generated %s file for OCR record %s',
            file_type,
            getattr(ocr_record, 'id', None),
        )
        return None


class EditedOCRExample(models.Model):
    ocr_record = models.ForeignKey(
        OCRRecord,
        on_delete=models.CASCADE,
        related_name='edits',
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='edited_examples',
    )
    edited_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    used = models.BooleanField(default=False)
    used_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f'Edit #{self.id} for OCRRecord #{self.ocr_record_id} by {self.user.username}'

    class Meta:
        indexes = [
            models.Index(fields=['used', 'created_at']),
        ]


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=OCRRecord)
def increment_storage_on_create(sender, instance, created, **kwargs):
    if created and instance.file_size and instance.user_id:
        UserProfile.objects.filter(user=instance.user).update(
            total_storage_used=F('total_storage_used') + instance.file_size,
        )


@receiver(post_delete, sender=OCRRecord)
def decrement_storage_and_delete_file(sender, instance, **kwargs):
    if instance.file_size and instance.user_id:
        profile = UserProfile.objects.filter(user=instance.user).first()
        if profile:
            profile.total_storage_used = max(0, profile.total_storage_used - instance.file_size)
            profile.save(update_fields=['total_storage_used'])

    if instance.image:
        instance.image.delete(save=False)

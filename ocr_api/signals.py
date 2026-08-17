"""Signal handlers for generated-file storage accounting and blob cleanup."""

from django.db.models import F
from django.db.models.signals import post_delete, post_save, pre_delete
from django.dispatch import receiver

from .models import GeneratedFile, OCRRecord, UserProfile


def _adjust_generated_storage(user_id, delta):
    """Apply ``delta`` bytes to generated + total storage counters."""
    if not user_id or not delta:
        return

    profile, _ = UserProfile.objects.get_or_create(user_id=user_id)

    if delta > 0:
        UserProfile.objects.filter(pk=profile.pk).update(
            generated_storage_used=F('generated_storage_used') + delta,
            total_storage_used=F('total_storage_used') + delta,
        )
        return

    profile.refresh_from_db()
    profile.generated_storage_used = max(0, profile.generated_storage_used + delta)
    profile.total_storage_used = max(0, profile.total_storage_used + delta)
    profile.save(update_fields=['generated_storage_used', 'total_storage_used'])


@receiver(post_save, sender=GeneratedFile)
def sync_generated_storage_on_save(sender, instance, created, **kwargs):
    """Increment or adjust ``generated_storage_used`` when a file is saved."""
    previous = getattr(instance, '_previous_file_size_bytes', 0) or 0
    current = instance.file_size_bytes or 0
    delta = current if created else current - previous
    _adjust_generated_storage(instance.user_id, delta)


@receiver(post_delete, sender=GeneratedFile)
def sync_generated_storage_on_delete(sender, instance, **kwargs):
    """Decrement storage counters and remove the blob after a row is deleted."""
    _adjust_generated_storage(instance.user_id, -(instance.file_size_bytes or 0))
    if instance.file:
        instance.file.delete(save=False)


@receiver(pre_delete, sender=OCRRecord)
def delete_generated_blobs_for_ocr_record(sender, instance, **kwargs):
    """Remove generated blobs before CASCADE deletes ``GeneratedFile`` rows."""
    for generated in instance.generated_files.all():
        if generated.file:
            generated.file.delete(save=False)

import os

from django.contrib.auth.models import User
from django.db import models
from django.db.models import F
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile',
    )
    total_storage_used = models.BigIntegerField(default=0)

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

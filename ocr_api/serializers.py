import os

from rest_framework import serializers

from .models import OCRRecord, UserProfile


class OCRRecordSerializer(serializers.ModelSerializer):
    user_email = serializers.EmailField(source='user.email', read_only=True)
    image_name = serializers.SerializerMethodField()
    image_size_bytes = serializers.SerializerMethodField()
    image_extension = serializers.SerializerMethodField()
    image_mime_type = serializers.SerializerMethodField()
    uploaded_at = serializers.SerializerMethodField()
    completed_at = serializers.SerializerMethodField()
    download_url = serializers.SerializerMethodField()

    class Meta:
        model = OCRRecord
        fields = [
            'id',
            'user',
            'image',
            'file_name',
            'file_size',
            'extracted_text',
            'status',
            'error_message',
            'document_type',
            'created_at',
            'user_email',
            'image_name',
            'image_size_bytes',
            'image_extension',
            'image_mime_type',
            'uploaded_at',
            'completed_at',
            'download_url',
        ]
        read_only_fields = [
            'user',
            'file_size',
            'file_name',
            'status',
            'extracted_text',
            'user_email',
            'image_name',
            'image_size_bytes',
            'image_extension',
            'image_mime_type',
            'uploaded_at',
            'completed_at',
            'download_url',
        ]

    def get_image_name(self, obj):
        return obj.file_name or (os.path.basename(obj.image.name) if obj.image and obj.image.name else None)

    def get_image_size_bytes(self, obj):
        return obj.file_size or 0

    def get_image_extension(self, obj):
        if not obj.image or not obj.image.name:
            return None
        return os.path.splitext(obj.image.name)[1].lower().lstrip('.')

    def get_image_mime_type(self, obj):
        if not obj.image or not obj.image.name:
            return None
        return getattr(obj.image, 'content_type', None) or 'application/octet-stream'

    def get_uploaded_at(self, obj):
        return obj.created_at.isoformat() if obj.created_at else None

    def get_completed_at(self, obj):
        return obj.updated_at.isoformat() if getattr(obj, 'updated_at', None) else None

    def get_download_url(self, obj):
        if not obj.id:
            return None
        return f'/api/download-ocr/{obj.id}/'


class StorageInfoSerializer(serializers.Serializer):
    user_id = serializers.IntegerField()
    username = serializers.CharField()
    email = serializers.EmailField()
    bytes_used = serializers.IntegerField()
    megabytes_used = serializers.FloatField()
    gigabytes_used = serializers.FloatField()

    @staticmethod
    def from_profile(profile):
        bytes_used = profile.total_storage_used
        return {
            'user_id': profile.user_id,
            'username': profile.user.username,
            'email': profile.user.email,
            'bytes_used': bytes_used,
            'megabytes_used': round(bytes_used / (1024 * 1024), 4),
            'gigabytes_used': round(bytes_used / (1024 * 1024 * 1024), 4),
        }

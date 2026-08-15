import os

from rest_framework import serializers

from .models import OCRRecord, UserProfile, EditedOCRExample


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


class UserListSerializer(serializers.ModelSerializer):
    class Meta:
        model = __import__('django.contrib.auth').contrib.auth.get_user_model()
        fields = ('id', 'username', 'email')


class UploadedFileListSerializer(serializers.ModelSerializer):
    uploader_name = serializers.CharField(source='user.username', read_only=True)
    uploader_email = serializers.EmailField(source='user.email', read_only=True)
    uploaded_at = serializers.DateTimeField(source='created_at', read_only=True)
    file_size = serializers.IntegerField(read_only=True)
    file_name = serializers.CharField(read_only=True)
    status = serializers.CharField(read_only=True)

    class Meta:
        model = OCRRecord
        fields = (
            'id',
            'file_name',
            'uploader_name',
            'uploader_email',
            'uploaded_at',
            'status',
            'file_size',
            'document_type',
            'image',
        )


class UserDetailSerializer(serializers.Serializer):
    user_id = serializers.IntegerField()
    username = serializers.CharField()
    email = serializers.EmailField()
    bytes_used = serializers.IntegerField()
    megabytes_used = serializers.FloatField()
    gigabytes_used = serializers.FloatField()
    total_files = serializers.IntegerField()
    total_completed = serializers.IntegerField()
    total_pending = serializers.IntegerField()
    total_failed = serializers.IntegerField()
    total_edits = serializers.IntegerField()
    total_used_edits = serializers.IntegerField()
    last_upload_at = serializers.DateTimeField(allow_null=True)

    @staticmethod
    def from_user(user):
        profile = getattr(user, 'profile', None)
        bytes_used = profile.total_storage_used if profile else 0
        ocr_qs = user.ocr_records.all()
        total_files = ocr_qs.count()
        total_completed = ocr_qs.filter(status=OCRRecord.STATUS_COMPLETED).count()
        total_pending = ocr_qs.filter(status=OCRRecord.STATUS_PENDING).count()
        total_failed = ocr_qs.filter(status=OCRRecord.STATUS_FAILED).count()
        total_edits = user.edited_examples.count()
        total_used_edits = user.edited_examples.filter(used=True).count()
        last_upload = ocr_qs.order_by('-created_at').values_list('created_at', flat=True).first()
        return {
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'bytes_used': bytes_used,
            'megabytes_used': round(bytes_used / (1024 * 1024), 4),
            'gigabytes_used': round(bytes_used / (1024 * 1024 * 1024), 4),
            'total_files': total_files,
            'total_completed': total_completed,
            'total_pending': total_pending,
            'total_failed': total_failed,
            'total_edits': total_edits,
            'total_used_edits': total_used_edits,
            'last_upload_at': last_upload,
        }


class EditedOCRExampleSerializer(serializers.ModelSerializer):
    user_email = serializers.EmailField(source='user.email', read_only=True)

    class Meta:
        model = EditedOCRExample
        fields = ('id', 'ocr_record', 'user', 'user_email', 'edited_text', 'created_at', 'used', 'used_at')
        read_only_fields = ('id', 'created_at', 'used', 'used_at', 'user_email')

    def create(self, validated_data):
        return super().create(validated_data)

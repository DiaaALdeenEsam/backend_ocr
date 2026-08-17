import os

from django.db.models import Count, Sum, Value
from django.db.models.functions import Coalesce
from rest_framework import serializers

from .models import GeneratedFile, OCRRecord, UserProfile, EditedOCRExample


def format_file_size(num_bytes):
    """Return a human-readable size such as ``180.0 KB`` (1024-based)."""
    value = float(num_bytes or 0)
    if value < 1024:
        return f'{int(value)} B'
    for unit in ('KB', 'MB', 'GB', 'TB'):
        value /= 1024.0
        if value < 1024 or unit == 'TB':
            return f'{value:.1f} {unit}'
    return f'{value:.1f} TB'


def _absolute_media_url(request, file_field):
    if not file_field or not getattr(file_field, 'name', None):
        return None
    try:
        url = file_field.url
    except ValueError:
        return None
    if request is None:
        return url
    try:
        return request.build_absolute_uri(url)
    except Exception:
        return url


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


class SourceImageSummarySerializer(serializers.ModelSerializer):
    """Nested source-image metadata shown on generated-file list/detail."""

    thumbnail_url = serializers.SerializerMethodField()
    image_url = serializers.SerializerMethodField()
    file_size_bytes = serializers.IntegerField(source='file_size', read_only=True)
    uploaded_at = serializers.DateTimeField(source='created_at', read_only=True)

    class Meta:
        model = OCRRecord
        fields = (
            'id',
            'file_name',
            'thumbnail_url',
            'image_url',
            'file_size_bytes',
            'status',
            'document_type',
            'uploaded_at',
        )

    def get_thumbnail_url(self, obj):
        request = self.context.get('request')
        return _absolute_media_url(request, obj.image)

    def get_image_url(self, obj):
        request = self.context.get('request')
        return _absolute_media_url(request, obj.image)


class GeneratedFileSerializer(serializers.ModelSerializer):
    file_size_human = serializers.SerializerMethodField()
    download_url = serializers.SerializerMethodField()
    source_image = SourceImageSummarySerializer(read_only=True)

    class Meta:
        model = GeneratedFile
        fields = (
            'id',
            'file_name',
            'file_type',
            'mime_type',
            'file_size_bytes',
            'file_size_human',
            'download_url',
            'created_at',
            'source_image',
        )
        read_only_fields = fields

    def get_file_size_human(self, obj):
        return format_file_size(obj.file_size_bytes)

    def get_download_url(self, obj):
        return f'/api/v1/generated-files/{obj.id}/download/'


class StorageTypeBreakdownSerializer(serializers.Serializer):
    file_type = serializers.CharField()
    count = serializers.IntegerField()
    bytes = serializers.IntegerField()
    megabytes = serializers.FloatField()


class StorageStatsSerializer(serializers.Serializer):
    """Aggregated generated-file storage, always including all four file types."""

    user_id = serializers.IntegerField()
    total_files = serializers.IntegerField()
    total_bytes = serializers.IntegerField()
    total_megabytes = serializers.FloatField()
    total_gigabytes = serializers.FloatField()
    by_file_type = StorageTypeBreakdownSerializer(many=True)

    FILE_TYPE_ORDER = (
        GeneratedFile.FileType.PDF,
        GeneratedFile.FileType.JSON,
        GeneratedFile.FileType.WORD,
        GeneratedFile.FileType.EXCEL,
    )

    @staticmethod
    def from_user(user, queryset=None):
        qs = queryset if queryset is not None else GeneratedFile.objects.filter(user=user)
        totals = qs.aggregate(
            total_bytes=Coalesce(Sum('file_size_bytes'), Value(0)),
            total_files=Count('id'),
        )
        total_bytes = int(totals['total_bytes'] or 0)
        total_files = int(totals['total_files'] or 0)

        by_type_rows = {
            row['file_type']: row
            for row in qs.values('file_type').annotate(
                count=Count('id'),
                bytes=Coalesce(Sum('file_size_bytes'), Value(0)),
            )
        }

        by_file_type = []
        for file_type in StorageStatsSerializer.FILE_TYPE_ORDER:
            row = by_type_rows.get(file_type, {})
            size_bytes = int(row.get('bytes') or 0)
            by_file_type.append({
                'file_type': file_type,
                'count': int(row.get('count') or 0),
                'bytes': size_bytes,
                'megabytes': round(size_bytes / (1024 * 1024), 4),
            })

        return {
            'user_id': user.id,
            'total_files': total_files,
            'total_bytes': total_bytes,
            'total_megabytes': round(total_bytes / (1024 * 1024), 4),
            'total_gigabytes': round(total_bytes / (1024 * 1024 * 1024), 4),
            'by_file_type': by_file_type,
        }

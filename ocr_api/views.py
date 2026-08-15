import io
import json
import gc
import logging
import os

from django.contrib.auth.models import User
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from PIL import Image
import torch
from rest_framework import status, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .models import OCRRecord, UserProfile, EditedOCRExample
from .serializers import OCRRecordSerializer, StorageInfoSerializer, EditedOCRExampleSerializer
from .serializers import UserDetailSerializer
from .ocr_engine import get_ocr_engine

logger = logging.getLogger(__name__)
MODEL_NAME = 'sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3'

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:
    arabic_reshaper = None
    get_display = None


def get_accessible_ocr_queryset(user):
    queryset = OCRRecord.objects.select_related('user')
    if user.is_staff or user.is_superuser:
        return queryset
    return queryset.filter(user=user)


def split_text_into_paragraphs(raw_text):
    text = (raw_text or '').strip()
    if not text:
        return []

    if '\n\n' in text:
        parts = text.split('\n\n')
    elif '\n' in text:
        parts = text.split('\n')
    else:
        return [{'index': 1, 'text': text}]

    paragraphs = []
    for part in parts:
        cleaned = part.strip()
        if cleaned:
            paragraphs.append({'index': len(paragraphs) + 1, 'text': cleaned})

    if not paragraphs:
        return []

    return paragraphs


def build_document_payload(ocr_record, request=None):
    if ocr_record.status != OCRRecord.STATUS_COMPLETED:
        return {'paragraphs': []}

    paragraphs = split_text_into_paragraphs(ocr_record.extracted_text)
    return {'paragraphs': paragraphs}


def build_metadata_payload(ocr_record, request=None):
    source_image = None
    if ocr_record.image and ocr_record.image.name:
        if request is not None:
            try:
                source_image = request.build_absolute_uri(ocr_record.image.url)
            except Exception:
                source_image = ocr_record.image.url
        else:
            source_image = ocr_record.image.url

    return {
        'created_at': ocr_record.created_at.isoformat() if ocr_record.created_at else None,
        'source_image': source_image,
        'model': MODEL_NAME,
    }


def serialize_ocr_record_response(ocr_record, request=None):
    raw_text = ocr_record.extracted_text if ocr_record.status == OCRRecord.STATUS_COMPLETED else None
    image_url = None
    if ocr_record.image and ocr_record.image.name:
        if request is not None:
            try:
                image_url = request.build_absolute_uri(ocr_record.image.url)
            except Exception:
                image_url = ocr_record.image.url
        else:
            image_url = ocr_record.image.url

    data = {
        'id': ocr_record.id,
        'status': ocr_record.status,
        'extracted_text': raw_text,
        'error_message': ocr_record.error_message,
        'document': build_document_payload(ocr_record, request),
        'metadata': build_metadata_payload(ocr_record, request),
        'image': {
            'name': ocr_record.file_name or (os.path.basename(ocr_record.image.name) if ocr_record.image and ocr_record.image.name else None),
            'size_bytes': ocr_record.file_size or 0,
            'uploaded_at': ocr_record.created_at.isoformat() if ocr_record.created_at else None,
            'extension': os.path.splitext((ocr_record.file_name or (ocr_record.image.name if ocr_record.image and ocr_record.image.name else '')))[1].lower().lstrip('.') if (ocr_record.file_name or (ocr_record.image.name if ocr_record.image and ocr_record.image.name else '')) else None,
            'mime_type': getattr(ocr_record.image, 'content_type', None) or 'application/octet-stream',
            'url': image_url,
        },
        'result': {
            'status': ocr_record.status,
            'completed_at': ocr_record.updated_at.isoformat() if getattr(ocr_record, 'updated_at', None) else None,
            'download': {
                'pdf_url': f'/api/download-ocr/{ocr_record.id}/?format=pdf' if ocr_record.status == OCRRecord.STATUS_COMPLETED else None,
                'json_url': f'/api/download-ocr/{ocr_record.id}/?format=json' if ocr_record.status == OCRRecord.STATUS_COMPLETED else None,
            },
        },
    }
    return data


def run_ocr_background(record_id):
    record = OCRRecord.objects.filter(pk=record_id).first()
    if not record:
        return

    # This function has been moved to a Celery task; keep a noop fallback for local calls
    try:
        logger.info('run_ocr_background invoked for record_id=%s but background processing is handled by Celery task.', record_id)
    except Exception:
        pass


class OCRRecordViewSet(viewsets.ModelViewSet):
    serializer_class = OCRRecordSerializer
    permission_classes = [IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser)

    def get_queryset(self):
        return get_accessible_ocr_queryset(self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class StorageInfoView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user = request.user
        is_admin = user.is_staff or user.is_superuser

        if is_admin:
            user_id = request.query_params.get('user_id')
            if user_id:
                profiles = UserProfile.objects.filter(user_id=user_id).select_related('user')
                if not profiles.exists():
                    return Response({'detail': 'User not found.'}, status=status.HTTP_404_NOT_FOUND)
                payload = StorageInfoSerializer.from_profile(profiles.first())
                return Response(payload, status=status.HTTP_200_OK)

            profiles = UserProfile.objects.select_related('user').order_by('user_id')
            payload = [StorageInfoSerializer.from_profile(profile) for profile in profiles]
            return Response(payload, status=status.HTTP_200_OK)

        profile, _ = UserProfile.objects.get_or_create(user=user)
        payload = StorageInfoSerializer.from_profile(profile)
        return Response(payload, status=status.HTTP_200_OK)


class UserDetailsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, user_id=None, *args, **kwargs):
        # allow admins to query other users; normal users can only query themselves
        if user_id:
            if not (request.user.is_staff or request.user.is_superuser):
                return Response({'detail': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)
            from django.contrib.auth.models import User
            user = User.objects.filter(pk=user_id).first()
            if not user:
                return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        else:
            user = request.user

        payload = UserDetailSerializer.from_user(user)
        return Response(payload, status=status.HTTP_200_OK)


class ProcessOCRView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser)
    from rest_framework.throttling import SimpleRateThrottle

    class UploadRateThrottle(SimpleRateThrottle):
        scope = 'uploads'
    throttle_classes = [UploadRateThrottle]

    def post(self, request, *args, **kwargs):
        serializer = OCRRecordSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        ocr_record = serializer.save(user=request.user)
        ocr_record.status = OCRRecord.STATUS_PENDING
        ocr_record.save(update_fields=['status'])
        
        # enqueue background processing via Celery task
        try:
            from .tasks import process_ocr_record
            process_ocr_record.delay(ocr_record.id)
        except Exception:
            logger.exception('Failed to enqueue OCR background task; falling back to synchronous processing')
            # fallback synchronous processing via Celery task apply
            try:
                process_ocr_record.apply(args=(ocr_record.id,))
            except Exception:
                logger.exception('Fallback synchronous OCR processing failed')

        return Response(
            {
                'id': ocr_record.id,
                'status': ocr_record.status,
                'message': 'OCR task accepted. Poll the status endpoint for updates.',
            },
            status=status.HTTP_202_ACCEPTED,
        )


class OCRStatusView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, pk, *args, **kwargs):
        ocr_record = get_object_or_404(get_accessible_ocr_queryset(request.user), pk=pk)
        data = serialize_ocr_record_response(ocr_record, request)
        return Response(data, status=status.HTTP_200_OK)

    def patch(self, request, pk, *args, **kwargs):
        ocr_record = get_object_or_404(get_accessible_ocr_queryset(request.user), pk=pk)

        if 'extracted_text' not in request.data:
            return Response(
                {'message': 'extracted_text is required.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        extracted_text = request.data.get('extracted_text')
        if extracted_text is None:
            extracted_text = ''
        if not isinstance(extracted_text, str):
            return Response(
                {'message': 'extracted_text must be a string.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Save the edited text as an audit/training example
        edit = EditedOCRExample.objects.create(
            ocr_record=ocr_record,
            user=request.user,
            edited_text=extracted_text,
        )

        # Also update the displayed extracted_text so users see their change immediately
        ocr_record.extracted_text = extracted_text
        ocr_record.save(update_fields=['extracted_text'])

        payload = EditedOCRExampleSerializer(edit).data
        payload.update({'message': 'Text updated and saved for training.', 'id': ocr_record.id, 'status': ocr_record.status, 'extracted_text': ocr_record.extracted_text})

        return Response(payload, status=status.HTTP_200_OK)

    def delete(self, request, pk, *args, **kwargs):
        ocr_record = get_object_or_404(get_accessible_ocr_queryset(request.user), pk=pk)
        ocr_record.delete()
        return Response({'message': 'Record and image deleted successfully'}, status=status.HTTP_200_OK)


class OCRHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        records = get_accessible_ocr_queryset(request.user).order_by('-created_at')
        payload = [serialize_ocr_record_response(record, request) for record in records]
        return Response(payload, status=status.HTTP_200_OK)


class OCRDownloadView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk, *args, **kwargs):
        ocr_record = get_object_or_404(get_accessible_ocr_queryset(request.user), pk=pk)

        if ocr_record.status != OCRRecord.STATUS_COMPLETED:
            return Response(
                {
                    'message': 'OCR record is not completed yet.',
                    'status': ocr_record.status,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        output_format = request.query_params.get('format', 'pdf').lower()
        if output_format not in ('pdf', 'json'):
            return Response(
                {'message': "Invalid format. Use 'pdf' or 'json'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if output_format == 'json':
            payload = {'text': ocr_record.extracted_text or ''}
            response = HttpResponse(
                json.dumps(payload, ensure_ascii=False, indent=2),
                content_type='application/json',
            )
            response['Content-Disposition'] = f'attachment; filename="ocr_result_{ocr_record.id}.json"'
            return response

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        left_margin = 40
        right_margin = 40
        top_margin = 50
        bottom_margin = 50
        line_spacing = 18
        font_name = 'ArialUnicode'
        font_size = 14

        font_path = r'C:\Windows\Fonts\arial.ttf'
        if not os.path.exists(font_path):
            return Response(
                {'message': f'Arabic font not found at {font_path}.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        try:
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, font_path))
        except Exception:
            return Response(
                {'message': 'Failed to load Arabic PDF font.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if not (arabic_reshaper and get_display):
            return Response(
                {'message': 'Arabic shaping libraries are missing. Install arabic-reshaper and python-bidi.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        def to_pdf_text(value):
            text = value or ''
            try:
                return get_display(arabic_reshaper.reshape(text))
            except Exception:
                return text

        def wrap_for_width(original_text, max_width):
            wrapped_lines = []
            paragraphs = original_text.splitlines() if original_text else ['']
            for paragraph in paragraphs:
                words = paragraph.split(' ')
                if not words:
                    wrapped_lines.append('')
                    continue

                current = []
                for word in words:
                    candidate_words = current + [word]
                    candidate_original = ' '.join(candidate_words).strip()
                    candidate_shaped = to_pdf_text(candidate_original)
                    candidate_width = pdfmetrics.stringWidth(candidate_shaped, font_name, font_size)

                    if candidate_width <= max_width or not current:
                        current = candidate_words
                    else:
                        wrapped_lines.append(' '.join(current).strip())
                        current = [word]

                wrapped_lines.append(' '.join(current).strip())
            return wrapped_lines

        max_text_width = width - left_margin - right_margin
        source_text = ocr_record.extracted_text or ''
        logical_lines = wrap_for_width(source_text, max_text_width)

        y = height - top_margin
        pdf.setFont(font_name, font_size)
        for logical_line in logical_lines:
            if y <= bottom_margin:
                pdf.showPage()
                pdf.setFont(font_name, font_size)
                y = height - top_margin

            shaped_line = to_pdf_text(logical_line)
            line_width = pdfmetrics.stringWidth(shaped_line, font_name, font_size)
            x = width - right_margin - line_width
            if x < left_margin:
                x = left_margin

            pdf.drawString(x, y, shaped_line)
            y -= line_spacing

        pdf.save()
        buffer.seek(0)

        response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="ocr_result_{ocr_record.id}.pdf"'
        return response


class MetricsView(APIView):
    # expose Prometheus metrics; no auth to allow monitoring systems to scrape
    authentication_classes = []
    permission_classes = []

    def get(self, request, *args, **kwargs):
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        except Exception:
            return Response({'detail': 'prometheus client not available'}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        data = generate_latest()
        return HttpResponse(data, content_type=CONTENT_TYPE_LATEST)

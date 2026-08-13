import io
import json
import gc
import logging
import os
import threading
import traceback
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from PIL import Image
import torch
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .models import OCRRecord
from .serializers import OCRRecordSerializer
from .ocr_engine import get_ocr_engine

logger = logging.getLogger(__name__)
MODEL_NAME = "sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3"

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:
    arabic_reshaper = None
    get_display = None


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
    data = {
        'id': ocr_record.id,
        'status': ocr_record.status,
        'extracted_text': raw_text,
        'error_message': ocr_record.error_message,
        'document': build_document_payload(ocr_record, request),
        'metadata': build_metadata_payload(ocr_record, request),
    }
    return data


def run_ocr_background(record_id):
    record = OCRRecord.objects.filter(pk=record_id).first()
    if not record:
        return

    try:
        # Step 1: upload/init
        record.status = OCRRecord.STATUS_UPLOADING
        record.error_message = None
        record.save(update_fields=['status', 'error_message'])

        # Step 2: full-image OCR processing (no line detection / cropping)
        record.status = OCRRecord.STATUS_PROCESSING
        record.error_message = None
        record.save(update_fields=['status', 'error_message'])

        ocr_engine = get_ocr_engine()

        try:
            extracted_text = (ocr_engine.predict(record.image.path) or '').strip()
            logger.info("Full-page OCR text for record_id=%s: '%s'", record.id, extracted_text)
        except Exception:
            logger.exception('Full-page OCR failed for record_id=%s', record.id)
            traceback.print_exc()
            extracted_text = ''
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        record.extracted_text = extracted_text
        record.status = OCRRecord.STATUS_COMPLETED
        record.error_message = None
        record.save(update_fields=['extracted_text', 'status', 'error_message'])
    except Exception as exc:
        record.error_message = str(exc)
        record.status = OCRRecord.STATUS_FAILED
        record.save(update_fields=['status', 'error_message'])

        logger.error('==================================================')
        logger.error('OCR BACKGROUND FAILURE for record_id=%s', record_id)
        logger.error('==================================================')
        logger.exception('Unhandled OCR background exception')
        traceback.print_exc()


class ProcessOCRView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = OCRRecordSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        ocr_record = serializer.save(status=OCRRecord.STATUS_UPLOADING)

        worker = threading.Thread(target=run_ocr_background, args=(ocr_record.id,), daemon=True)
        worker.start()

        return Response(
            {
                'id': ocr_record.id,
                'status': ocr_record.status,
                'message': 'OCR task accepted. Poll the status endpoint for updates.'
            },
            status=status.HTTP_202_ACCEPTED,
        )


class OCRStatusView(APIView):
    def get(self, request, pk, *args, **kwargs):
        ocr_record = get_object_or_404(OCRRecord, pk=pk)
        data = serialize_ocr_record_response(ocr_record, request)
        return Response(data, status=status.HTTP_200_OK)

    def patch(self, request, pk, *args, **kwargs):
        ocr_record = get_object_or_404(OCRRecord, pk=pk)

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

        ocr_record.extracted_text = extracted_text
        ocr_record.save(update_fields=['extracted_text'])

        return Response(
            {
                'message': 'Text updated successfully',
                'id': ocr_record.id,
                'status': ocr_record.status,
                'extracted_text': ocr_record.extracted_text,
            },
            status=status.HTTP_200_OK,
        )

    def delete(self, request, pk, *args, **kwargs):
        ocr_record = get_object_or_404(OCRRecord, pk=pk)
        ocr_record.image.delete(save=False)
        ocr_record.delete()
        return Response({'message': 'Record and image deleted successfully'}, status=status.HTTP_200_OK)


class OCRHistoryView(APIView):
    def get(self, request, *args, **kwargs):
        records = OCRRecord.objects.all().order_by('-created_at')
        payload = [serialize_ocr_record_response(record, request) for record in records]
        return Response(payload, status=status.HTTP_200_OK)


class OCRDownloadView(APIView):
    def get(self, request, pk, *args, **kwargs):
        ocr_record = get_object_or_404(OCRRecord, pk=pk)

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
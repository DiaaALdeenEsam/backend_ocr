from django.urls import path
from .views import ProcessOCRView, OCRStatusView, OCRHistoryView, OCRDownloadView

app_name = 'ocr_api'

urlpatterns = [
    path('process-ocr/', ProcessOCRView.as_view(), name='process-ocr'),
    path('ocr-status/<int:pk>/', OCRStatusView.as_view(), name='ocr-status'),
    path('ocr-history/', OCRHistoryView.as_view(), name='ocr-history'),
    path('download-ocr/<int:pk>/', OCRDownloadView.as_view(), name='ocr-download'),
]
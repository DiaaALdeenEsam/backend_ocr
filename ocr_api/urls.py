from django.urls import path
from rest_framework.routers import DefaultRouter

from .auth_views import LoginView, LogoutView, SignupView
from .exporters import OCRExportDocxView, OCRExportPdfView, OCRExportXlsxView
from .views import (
    OCRDownloadView,
    OCRHistoryView,
    OCRHistorySearchView,
    OCRRecordViewSet,
    OCRStatusView,
    ProcessOCRView,
    StorageInfoView,
    UploadedFilesView,
    UserDetailsView,
    UserListView,
)

app_name = 'ocr_api'

router = DefaultRouter()
router.register(r'ocr-records', OCRRecordViewSet, basename='ocr-record')

urlpatterns = [
    path('auth/signup/', SignupView.as_view(), name='auth-signup'),
    path('auth/login/', LoginView.as_view(), name='auth-login'),
    path('auth/logout/', LogoutView.as_view(), name='auth-logout'),
    path('storage-info/', StorageInfoView.as_view(), name='storage-info'),
    path('users/', UserListView.as_view(), name='user-list'),
    path('uploaded-files/', UploadedFilesView.as_view(), name='uploaded-files'),
    path('user-details/', UserDetailsView.as_view(), name='user-details'),
    path('user-details/<int:user_id>/', UserDetailsView.as_view(), name='user-details-admin'),
    path('process-ocr/', ProcessOCRView.as_view(), name='process-ocr'),
    path('ocr-status/<int:pk>/', OCRStatusView.as_view(), name='ocr-status'),
    path('ocr-history/', OCRHistoryView.as_view(), name='ocr-history'),
    path('ocr-history/search/', OCRHistorySearchView.as_view(), name='ocr-history-search'),
    path('download-ocr/<int:pk>/', OCRDownloadView.as_view(), name='ocr-download'),
    path('export-ocr/<int:pk>/pdf/', OCRExportPdfView.as_view(), name='export-ocr-pdf'),
    path('export-ocr/<int:pk>/docx/', OCRExportDocxView.as_view(), name='export-ocr-docx'),
    path('export-ocr/<int:pk>/xlsx/', OCRExportXlsxView.as_view(), name='export-ocr-xlsx'),
] + router.urls

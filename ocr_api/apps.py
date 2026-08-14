from django.apps import AppConfig


class OcrApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ocr_api'

    def ready(self):
        from . import models  # noqa: F401

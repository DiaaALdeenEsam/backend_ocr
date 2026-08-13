from rest_framework import serializers
from .models import OCRRecord

class OCRRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = OCRRecord
        fields = ['id', 'image', 'extracted_text', 'error_message', 'created_at']
        read_only_fields = ['extracted_text', 'error_message', 'created_at']

    def to_representation(self, instance):
        data = super().to_representation(instance)
        return data
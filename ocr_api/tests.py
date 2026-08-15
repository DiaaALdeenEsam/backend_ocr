from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
import io
from PIL import Image
from rest_framework.test import APITestCase
from unittest.mock import patch

from ocr_api.models import OCRRecord

User = get_user_model()


class UserListEndpointTests(APITestCase):
    def test_admin_can_list_users_with_username_and_email(self):
        User.objects.create_user(
            username='adminuser',
            email='admin@example.com',
            password='StrongPass123!',
            is_staff=True,
            is_superuser=True,
        )
        User.objects.create_user(
            username='alice',
            email='alice@example.com',
            password='StrongPass123!',
        )
        User.objects.create_user(
            username='bob',
            email='bob@example.com',
            password='StrongPass123!',
        )

        self.client.force_authenticate(user=User.objects.get(username='adminuser'))
        response = self.client.get(reverse('ocr_api:user-list'))

        self.assertEqual(response.status_code, 200, response.data)
        self.assertIsInstance(response.data, list)
        self.assertTrue(any(item['username'] == 'alice' and item['email'] == 'alice@example.com' for item in response.data))
        self.assertTrue(any(item['username'] == 'bob' and item['email'] == 'bob@example.com' for item in response.data))

    def test_non_admin_cannot_list_users(self):
        User.objects.create_user(
            username='normaluser',
            email='normal@example.com',
            password='StrongPass123!',
        )

        self.client.force_authenticate(user=User.objects.get(username='normaluser'))
        response = self.client.get(reverse('ocr_api:user-list'))

        self.assertEqual(response.status_code, 403, response.data)
        self.assertEqual(response.data['detail'], 'Forbidden')


class UploadedFilesEndpointTests(APITestCase):
    def test_admin_can_list_uploaded_files_with_uploader_and_time(self):
        admin = User.objects.create_user(
            username='adminfiles',
            email='adminfiles@example.com',
            password='StrongPass123!',
            is_staff=True,
            is_superuser=True,
        )
        uploader = User.objects.create_user(
            username='uploader1',
            email='uploader1@example.com',
            password='StrongPass123!',
        )
        OCRRecord.objects.create(
            user=uploader,
            image=SimpleUploadedFile('sample.png', b'fake-image-data', content_type='image/png'),
            file_name='sample.png',
            file_size=123,
            status=OCRRecord.STATUS_COMPLETED,
            extracted_text='hello',
        )

        self.client.force_authenticate(user=admin)
        response = self.client.get(reverse('ocr_api:uploaded-files'))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(
            item['uploader_name'] == 'uploader1' and item['uploader_email'] == 'uploader1@example.com' and item['file_name'] == 'sample.png'
            for item in response.data
        ))

    def test_non_admin_cannot_list_uploaded_files(self):
        user = User.objects.create_user(
            username='regularuser2',
            email='regular2@example.com',
            password='StrongPass123!',
        )
        self.client.force_authenticate(user=user)

        response = self.client.get(reverse('ocr_api:uploaded-files'))

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data['detail'], 'Forbidden')


class OCRHistorySearchEndpointTests(APITestCase):
    def test_user_can_search_own_history_by_ocr_output_text(self):
        user = User.objects.create_user(
            username='historyuser',
            email='history@example.com',
            password='StrongPass123!',
        )
        other_user = User.objects.create_user(
            username='otherhistory',
            email='otherhistory@example.com',
            password='StrongPass123!',
        )

        OCRRecord.objects.create(
            user=user,
            image=SimpleUploadedFile('invoice.png', b'img-1', content_type='image/png'),
            file_name='invoice_2026.png',
            file_size=10,
            status=OCRRecord.STATUS_COMPLETED,
            extracted_text='Client name: Ahmed Hassan',
        )
        OCRRecord.objects.create(
            user=other_user,
            image=SimpleUploadedFile('secret.png', b'img-2', content_type='image/png'),
            file_name='secret_notes.png',
            file_size=12,
            status=OCRRecord.STATUS_COMPLETED,
            extracted_text='Ahmed private record',
        )

        self.client.force_authenticate(user=user)
        response = self.client.get(reverse('ocr_api:ocr-history-search'), {'q': 'Ahmed'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['count'], 1)
        self.assertEqual(response.data['results'][0]['image']['name'], 'invoice.png')

    def test_user_can_search_own_history_by_file_name(self):
        user = User.objects.create_user(
            username='historyuser2',
            email='history2@example.com',
            password='StrongPass123!',
        )
        OCRRecord.objects.create(
            user=user,
            image=SimpleUploadedFile('contract.png', b'img-3', content_type='image/png'),
            file_name='employment_contract.png',
            file_size=20,
            status=OCRRecord.STATUS_COMPLETED,
            extracted_text='Signed contract details',
        )

        self.client.force_authenticate(user=user)
        response = self.client.get(reverse('ocr_api:ocr-history-search'), {'q': 'contract'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['count'], 1)
        self.assertEqual(response.data['results'][0]['image']['name'], 'contract.png')

    def test_search_requires_query_parameter(self):
        user = User.objects.create_user(
            username='historyuser3',
            email='history3@example.com',
            password='StrongPass123!',
        )
        self.client.force_authenticate(user=user)

        response = self.client.get(reverse('ocr_api:ocr-history-search'))

        self.assertEqual(response.status_code, 400)
        self.assertIn('Please provide a search query', response.data['detail'])


class ProcessOCREndpointTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='ocruser',
            email='ocruser@example.com',
            password='StrongPass123!',
        )
        self.client.force_authenticate(user=self.user)

    def test_missing_image_returns_400_not_500(self):
        response = self.client.post(reverse('ocr_api:process-ocr'), data={}, format='multipart')

        self.assertEqual(response.status_code, 400)
        self.assertIn('image', response.data)

    @patch('ocr_api.views.current_app.send_task')
    def test_valid_image_is_accepted(self, mock_send_task):
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), color='white').save(buffer, format='PNG')
        image = SimpleUploadedFile('sample.png', buffer.getvalue(), content_type='image/png')
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.post(
                reverse('ocr_api:process-ocr'),
                data={'image': image},
                format='multipart',
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.data['status'], OCRRecord.STATUS_PENDING)
        self.assertIn('id', response.data)
        mock_send_task.assert_called_once_with(
            'ocr_api.tasks.process_ocr_record',
            args=[response.data['id']],
            countdown=1,
        )

    @patch('ocr_api.views.current_app.send_task', side_effect=RuntimeError('broker down'))
    def test_queue_failure_still_returns_pending_immediately(self, mock_send_task):
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), color='white').save(buffer, format='PNG')
        image = SimpleUploadedFile('sample.png', buffer.getvalue(), content_type='image/png')

        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.post(
                reverse('ocr_api:process-ocr'),
                data={'image': image},
                format='multipart',
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.data['status'], OCRRecord.STATUS_PENDING)
        self.assertIn('id', response.data)
        mock_send_task.assert_called_once_with(
            'ocr_api.tasks.process_ocr_record',
            args=[response.data['id']],
            countdown=1,
        )


class MetricsEndpointTests(APITestCase):
    def test_admin_can_access_metrics_and_sees_extra_summary(self):
        admin = User.objects.create_user(
            username='adminmetrics',
            email='adminmetrics@example.com',
            password='StrongPass123!',
            is_staff=True,
            is_superuser=True,
        )
        self.client.force_authenticate(user=admin)

        response = self.client.get('/metrics/')

        self.assertEqual(response.status_code, 200)
        self.assertIn('ocr_total_users', response.content.decode())
        self.assertIn('ocr_total_records', response.content.decode())
        self.assertIn('ocr_total_completed', response.content.decode())

    def test_non_admin_cannot_access_metrics(self):
        user = User.objects.create_user(
            username='regularuser',
            email='regular@example.com',
            password='StrongPass123!',
        )
        self.client.force_authenticate(user=user)

        response = self.client.get('/metrics/')

        self.assertEqual(response.status_code, 403)
        self.assertIn('Access denied: admin privileges required.', str(response.data))

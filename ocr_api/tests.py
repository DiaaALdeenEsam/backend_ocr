from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APITestCase

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

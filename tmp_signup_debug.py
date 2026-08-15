import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'core.settings'
import django
django.setup()
from django.test import Client
c = Client()
r = c.post('/api/auth/signup/', {'username':'newuser','email':'newuser@example.com','password':'StrongPass123!','password_confirm':'StrongPass123!'}, content_type='application/json')
print('status=', r.status_code)
print(r.content.decode())

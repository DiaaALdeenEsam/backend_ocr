from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

User = get_user_model()


class SignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password', 'password_confirm')

    def validate(self, attrs):
        # uniqueness checks
        username = attrs.get('username')
        email = attrs.get('email')
        if username and User.objects.filter(username__iexact=username).exists():
            raise serializers.ValidationError({'username': 'A user with that username already exists.'})
        if email and User.objects.filter(email__iexact=email).exists():
            raise serializers.ValidationError({'email': 'A user with that email already exists.'})
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({'password_confirm': 'Passwords do not match.'})
        return attrs

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        return User.objects.create_user(**validated_data)

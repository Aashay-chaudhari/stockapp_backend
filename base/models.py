from django.db import models
from django.contrib import admin
from django.utils import timezone


# Create your models here.

class Stock(models.Model):
    name = models.CharField(max_length=200)
    added = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class UserModel(models.Model):
    email = models.CharField(max_length=200, unique=True)
    password = models.CharField(max_length=200)

    def __str__(self):
        return self.email


class ActiveUser(models.Model):
    user_email = models.CharField(max_length=200)
    access_token = models.CharField(max_length=200)
    last_active = models.DateTimeField()

    def __str__(self):
        return self.user_email

    @classmethod
    def delete_inactive_users(cls):
        # Calculate the threshold time (1 hour ago)
        threshold_time = timezone.now() - timezone.timedelta(hours=1)

        # Delete instances where last_active is older than the threshold time
        cls.objects.filter(last_active__lt=threshold_time).delete()

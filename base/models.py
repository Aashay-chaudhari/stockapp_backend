from django.db import models

# Create your models here.

class Stock(models.Model):
    name = models.CharField(max_length=200)
    added = models.DateTimeField(auto_now_add=True)

class UserModel(models.Model):
    name = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
# Generated by Django 4.2.2 on 2023-10-25 21:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0003_rename_name_usermodel_email'),
    ]

    operations = [
        migrations.CreateModel(
            name='ActiveUser',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_email', models.CharField(max_length=200)),
                ('access_token', models.CharField(max_length=200)),
                ('last_active', models.DateTimeField()),
            ],
        ),
    ]
from django.contrib import admin
from base.models import UserModel, Stock, ActiveUser


# Register the model
admin.site.register(UserModel)

admin.site.register(Stock)

admin.site.register(ActiveUser)

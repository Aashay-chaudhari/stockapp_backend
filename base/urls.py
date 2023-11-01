from django.urls import path
from base import views

urlpatterns = [
    # path('', views.getModelData),
    path('getUserData/', views.get_user_data),
    path('add/', views.add_stock),
    path('addUser/', views.add_user),
    path('checkLogin/', views.check_login),
    path('getStockData/', views.get_stock_data),
    path('predict/', views.predict),
    path('getModelData/', views.get_model_data),
    path('trainModel/', views.trainModel_with_new_scaling),
    path('show_similar/', views.show_similar),
]


from django.urls import path
from . import views

urlpatterns = [
    # path('', views.getModelData),
    path('getUserData/', views.getUserData),
    path('add/', views.addStock),
    path('addUser/', views.addUser),
    path('checkLogin/', views.checkLogin),
    path('getStockData/', views.getStockData),
    path('predict/', views.predict),
    path('getModelData/', views.getModelData),
    path('trainModel/', views.trainModel_with_new_scaling),
    path('show_similar/', views.show_similar),
]


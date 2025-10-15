from django.urls import path
from . import views

app_name = 'ml_predicciones'

urlpatterns = [
    path('', views.dashboard_ml, name='dashboard'),
    path('predecir/<str:placa>/', views.predecir_vehiculo, name='predecir'),
    path('estadisticas/', views.estadisticas_ml, name='estadisticas'),
]

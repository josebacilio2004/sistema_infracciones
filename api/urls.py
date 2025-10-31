from django.urls import path
from . import views

urlpatterns = [
    # Endpoints para predicciones desde Google Colab
    path('prediccion/reincidencia/', views.predecir_reincidencia, name='predecir_reincidencia'),
    path('prediccion/accidente/', views.predecir_accidente, name='predecir_accidente'),
    path('prediccion/riesgo-conductor/', views.predecir_riesgo_conductor, name='predecir_riesgo_conductor'),
    
    # Endpoints para obtener datos de entrenamiento
    path('datos/infracciones/', views.obtener_datos_infracciones, name='obtener_datos_infracciones'),
    path('datos/vehiculos/', views.obtener_datos_vehiculos, name='obtener_datos_vehiculos'),
    
    # Endpoint para registrar infracción detectada
    path('infraccion/registrar/', views.registrar_infraccion, name='registrar_infraccion'),
    
    # Endpoint de prueba
    path('test/', views.api_test, name='api_test'),
]

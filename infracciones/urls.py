from django.urls import path
from . import views

app_name = 'infracciones'

urlpatterns = [
    path('', views.lista_infracciones, name='lista'),
    path('<int:pk>/', views.detalle_infraccion, name='detalle'),
    path('estadisticas/', views.estadisticas, name='estadisticas'),
]

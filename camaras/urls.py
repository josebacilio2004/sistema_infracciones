from django.urls import path
from . import views

urlpatterns = [
    path("", views.lista_camaras, name="camaras_lista"),
    path("crear/", views.crear_camara, name="camaras_crear"),
    path("editar/<int:pk>/", views.editar_camara, name="camaras_editar"),
    path("eliminar/<int:pk>/", views.eliminar_camara, name="camaras_eliminar"),
]

from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="dashboard_home"),  # /dashboard/
    path("video-feed/", views.video_feed, name="video_feed"),  # /dashboard/video-feed/
    path("api/detecciones/", views.api_detecciones, name="api_detecciones"),  # /dashboard/api/detecciones/
    path("api/procesar-frame/", views.procesar_frame, name="procesar_frame"),  # /dashboard/api/procesar-frame/
    path("api/camaras-disponibles/", views.listar_camaras_disponibles, name="listar_camaras"),  # /dashboard/api/camaras-disponibles/
    path("api/cambiar-camara/", views.cambiar_camara, name="cambiar_camara"),  # /dashboard/api/cambiar-camara/
    path("api/estadisticas-tiempo-real/", views.estadisticas_tiempo_real, name="estadisticas_tiempo_real"),  # /dashboard/api/estadisticas-tiempo-real/
    path("api/consultar-sunarp/", views.consultar_sunarp, name="consultar_sunarp"),  # /dashboard/api/consultar-sunarp/
    path("probar-camara-rtsp/", views.probar_camara_rtsp, name="probar_camara_rtsp"),  # /dashboard/probar-camara-rtsp/
]

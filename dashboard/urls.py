from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="dashboard_home"),
    path("video-feed/", views.video_feed, name="video_feed"),
    path("api/detecciones/", views.api_detecciones, name="api_detecciones"),
]
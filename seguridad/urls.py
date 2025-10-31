from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from dashboard.views import presentacion_sistema

urlpatterns = [
    path("", presentacion_sistema, name="inicio"),  # Ruta principal pública con presentación
    path("login/", auth_views.LoginView.as_view(template_name='auth/login.html'), name="login"),  # Login
    path("accounts/profile/", lambda request: redirect("dashboard_home")),  # Agregar ruta para /accounts/profile/ que redirija al dashboard
    path("logout/", auth_views.LogoutView.as_view(next_page='inicio'), name="logout"),  # Logout
    path("admin/", admin.site.urls),
    path("dashboard/", include("dashboard.urls")),
    path("camaras/", include("camaras.urls")),
    path("infracciones/", include("infracciones.urls")),
    path("ml-predicciones/", include("ml_predicciones.urls")),
    path("api/", include("api.urls")),
]
    
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

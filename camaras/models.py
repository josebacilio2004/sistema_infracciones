from django.db import models

class Camara(models.Model):
    ubicacion = models.CharField(max_length=200)
    ip = models.GenericIPAddressField(protocol="both", unpack_ipv4=False, null=True, blank=True)
    descripcion = models.TextField(null=True, blank=True)
    activa = models.BooleanField(default=True, help_text="Indica si la cámara está activa y operativa")
    fecha_instalacion = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    ultima_conexion = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "Cámara"
        verbose_name_plural = "Cámaras"

    def __str__(self):
        return f"{self.ubicacion} ({self.ip})"
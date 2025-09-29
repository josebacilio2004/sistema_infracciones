from django.db import models

class Camara(models.Model):
    ubicacion = models.CharField(max_length=200)
    ip = models.GenericIPAddressField(protocol="both", unpack_ipv4=False, null=True, blank=True)
    descripcion = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.ubicacion} ({self.ip})"

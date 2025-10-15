from django.db import models
from django.utils import timezone

class Camara(models.Model):
    ubicacion = models.CharField(max_length=200)
    ip = models.GenericIPAddressField(protocol="both", unpack_ipv4=False, null=True, blank=True)
    descripcion = models.TextField(null=True, blank=True)
    activa = models.BooleanField(default=True, help_text="Indica si la cámara está activa y operativa")
    fecha_instalacion = models.DateTimeField(default=timezone.now, null=True, blank=True)
    ultima_conexion = models.DateTimeField(null=True, blank=True)
    
    tipo_fuente = models.CharField(
        max_length=20,
        choices=[
            ('WEBCAM', 'Webcam Local'),
            ('IP', 'Cámara IP'),
            ('VIDEO', 'Archivo de Video'),
            ('IRIUN', 'IriunWebcam')
        ],
        default='WEBCAM',
        help_text="Tipo de fuente de video"
    )
    url_stream = models.CharField(
        max_length=500,
        null=True,
        blank=True,
        help_text="URL para cámara IP o IriunWebcam (ej: http://192.168.1.100:8080/video)"
    )
    indice_webcam = models.IntegerField(
        default=0,
        help_text="Índice de la webcam local (0, 1, 2, etc.)"
    )
    ruta_video = models.CharField(
        max_length=500,
        null=True,
        blank=True,
        help_text="Ruta del archivo de video para pruebas"
    )

    class Meta:
        verbose_name = "Cámara"
        verbose_name_plural = "Cámaras"

    def __str__(self):
        return f"{self.ubicacion} ({self.get_tipo_fuente_display()})"
    
    def obtener_fuente_video(self):
        """Retorna la fuente de video según el tipo configurado"""
        if self.tipo_fuente == 'WEBCAM':
            return self.indice_webcam
        elif self.tipo_fuente == 'IP' or self.tipo_fuente == 'IRIUN':
            return self.url_stream
        elif self.tipo_fuente == 'VIDEO':
            return self.ruta_video
        return 0

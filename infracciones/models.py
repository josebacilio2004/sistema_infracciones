from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from camaras.models import Camara

class TipoInfraccion(models.Model):
    """Catálogo de tipos de infracciones detectables"""
    codigo = models.CharField(max_length=20, unique=True)
    nombre = models.CharField(max_length=200)
    descripcion = models.TextField()
    monto_multa = models.DecimalField(max_digits=10, decimal_places=2)
    puntos_licencia = models.IntegerField(default=0)
    gravedad = models.CharField(
        max_length=20,
        choices=[
            ('LEVE', 'Leve'),
            ('MODERADA', 'Moderada'),
            ('GRAVE', 'Grave'),
            ('MUY_GRAVE', 'Muy Grave')
        ],
        default='LEVE'
    )
    activo = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = "Tipo de Infracción"
        verbose_name_plural = "Tipos de Infracciones"
    
    def __str__(self):
        return f"{self.codigo} - {self.nombre}"


class Vehiculo(models.Model):
    """Información de vehículos detectados"""
    placa = models.CharField(max_length=20, unique=True, db_index=True)
    marca = models.CharField(max_length=100, null=True, blank=True)
    modelo = models.CharField(max_length=100, null=True, blank=True)
    color = models.CharField(max_length=50, null=True, blank=True)
    anio = models.IntegerField(null=True, blank=True)
    tipo_vehiculo = models.CharField(
        max_length=50,
        choices=[
            ('AUTO', 'Automóvil'),
            ('MOTO', 'Motocicleta'),
            ('CAMION', 'Camión'),
            ('BUS', 'Autobús'),
            ('OTRO', 'Otro')
        ],
        default='AUTO'
    )
    propietario_nombre = models.CharField(max_length=200, null=True, blank=True)
    propietario_documento = models.CharField(max_length=50, null=True, blank=True)
    propietario_telefono = models.CharField(max_length=20, null=True, blank=True)
    propietario_email = models.EmailField(null=True, blank=True)
    reportado_robado = models.BooleanField(default=False)
    fecha_registro = models.DateTimeField(default=timezone.now)
    
    class Meta:
        verbose_name = "Vehículo"
        verbose_name_plural = "Vehículos"
    
    def __str__(self):
        return f"{self.placa} - {self.marca} {self.modelo}"
    
    def total_infracciones(self):
        return self.infracciones.count()
    
    def infracciones_ultimos_30_dias(self):
        from datetime import timedelta
        fecha_limite = timezone.now() - timedelta(days=30)
        return self.infracciones.filter(fecha_hora__gte=fecha_limite).count()


class Infraccion(models.Model):
    """Registro de infracciones detectadas por IA"""
    vehiculo = models.ForeignKey(Vehiculo, on_delete=models.CASCADE, related_name='infracciones')
    tipo_infraccion = models.ForeignKey(TipoInfraccion, on_delete=models.PROTECT)
    camara = models.ForeignKey(Camara, on_delete=models.SET_NULL, null=True)
    
    fecha_hora = models.DateTimeField(default=timezone.now, db_index=True)
    ubicacion = models.CharField(max_length=300)
    latitud = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitud = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    
    # Datos específicos según tipo de infracción
    velocidad_detectada = models.IntegerField(null=True, blank=True, help_text="km/h")
    velocidad_maxima = models.IntegerField(null=True, blank=True, help_text="km/h")
    tiempo_luz_roja = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True, help_text="segundos")
    
    # Evidencia
    imagen_principal = models.ImageField(upload_to='infracciones/imagenes/', null=True, blank=True)
    imagen_placa = models.ImageField(upload_to='infracciones/placas/', null=True, blank=True)
    video_evidencia = models.FileField(upload_to='infracciones/videos/', null=True, blank=True)
    
    # Confianza del modelo de IA
    confianza_deteccion = models.DecimalField(max_digits=5, decimal_places=2, help_text="Porcentaje de confianza del modelo")
    modelo_ia_version = models.CharField(max_length=50, default='v1.0')
    
    # Estado de la infracción
    estado = models.CharField(
        max_length=20,
        choices=[
            ('DETECTADA', 'Detectada'),
            ('VERIFICADA', 'Verificada'),
            ('NOTIFICADA', 'Notificada'),
            ('PAGADA', 'Pagada'),
            ('IMPUGNADA', 'Impugnada'),
            ('ANULADA', 'Anulada')
        ],
        default='DETECTADA',
        db_index=True
    )
    
    verificada_por = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='infracciones_verificadas')
    fecha_verificacion = models.DateTimeField(null=True, blank=True)
    notas_verificacion = models.TextField(null=True, blank=True)
    
    fecha_notificacion = models.DateTimeField(null=True, blank=True)
    fecha_pago = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Infracción"
        verbose_name_plural = "Infracciones"
        ordering = ['-fecha_hora']
        indexes = [
            models.Index(fields=['fecha_hora', 'estado']),
            models.Index(fields=['vehiculo', 'fecha_hora']),
        ]
    
    def __str__(self):
        return f"{self.vehiculo.placa} - {self.tipo_infraccion.nombre} - {self.fecha_hora.strftime('%Y-%m-%d %H:%M')}"


class PerfilConductor(models.Model):
    """Perfil de riesgo del conductor basado en historial"""
    vehiculo = models.OneToOneField(Vehiculo, on_delete=models.CASCADE, related_name='perfil')
    
    # Estadísticas
    total_infracciones = models.IntegerField(default=0)
    infracciones_luz_roja = models.IntegerField(default=0)
    infracciones_velocidad = models.IntegerField(default=0)
    infracciones_graves = models.IntegerField(default=0)
    
    # Puntuación de riesgo (calculada por ML)
    puntuacion_riesgo = models.DecimalField(max_digits=5, decimal_places=2, default=0.0, help_text="0-100")
    nivel_riesgo = models.CharField(
        max_length=20,
        choices=[
            ('BAJO', 'Bajo'),
            ('MEDIO', 'Medio'),
            ('ALTO', 'Alto'),
            ('CRITICO', 'Crítico')
        ],
        default='BAJO'
    )
    
    # Predicciones
    probabilidad_reincidencia = models.DecimalField(max_digits=5, decimal_places=2, default=0.0, help_text="Porcentaje")
    probabilidad_accidente = models.DecimalField(max_digits=5, decimal_places=2, default=0.0, help_text="Porcentaje")
    
    ultima_actualizacion = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Perfil de Conductor"
        verbose_name_plural = "Perfiles de Conductores"
    
    def __str__(self):
        return f"Perfil {self.vehiculo.placa} - Riesgo: {self.nivel_riesgo}"


class PrediccionAccidente(models.Model):
    """Predicciones de posibles accidentes en zonas específicas"""
    ubicacion = models.CharField(max_length=300)
    latitud = models.DecimalField(max_digits=9, decimal_places=6)
    longitud = models.DecimalField(max_digits=9, decimal_places=6)
    
    fecha_prediccion = models.DateTimeField(default=timezone.now)
    periodo_prediccion = models.CharField(
        max_length=50,
        choices=[
            ('PROXIMA_HORA', 'Próxima Hora'),
            ('PROXIMO_DIA', 'Próximo Día'),
            ('PROXIMA_SEMANA', 'Próxima Semana'),
            ('PROXIMO_MES', 'Próximo Mes')
        ]
    )
    
    probabilidad = models.DecimalField(max_digits=5, decimal_places=2, help_text="Porcentaje")
    factores_riesgo = models.JSONField(help_text="Factores que contribuyen al riesgo")
    
    infracciones_historicas = models.IntegerField(default=0)
    accidentes_historicos = models.IntegerField(default=0)
    
    modelo_version = models.CharField(max_length=50, default='v1.0')
    
    class Meta:
        verbose_name = "Predicción de Accidente"
        verbose_name_plural = "Predicciones de Accidentes"
        ordering = ['-probabilidad', '-fecha_prediccion']
    
    def __str__(self):
        return f"{self.ubicacion} - {self.probabilidad}% - {self.periodo_prediccion}"


class EventoDeteccion(models.Model):
    """Log de eventos de detección en tiempo real"""
    camara = models.ForeignKey(Camara, on_delete=models.CASCADE, related_name='eventos')
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    
    tipo_evento = models.CharField(
        max_length=50,
        choices=[
            ('VEHICULO_DETECTADO', 'Vehículo Detectado'),
            ('PLACA_RECONOCIDA', 'Placa Reconocida'),
            ('INFRACCION_DETECTADA', 'Infracción Detectada'),
            ('ERROR_DETECCION', 'Error de Detección'),
            ('CAMARA_OFFLINE', 'Cámara Offline')
        ]
    )
    
    datos_evento = models.JSONField(help_text="Datos adicionales del evento")
    imagen_frame = models.ImageField(upload_to='eventos/frames/', null=True, blank=True)
    
    class Meta:
        verbose_name = "Evento de Detección"
        verbose_name_plural = "Eventos de Detección"
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.camara.ubicacion} - {self.tipo_evento} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

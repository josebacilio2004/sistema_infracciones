from django.db import models

class ModeloEntrenamiento(models.Model):
    """Registro de modelos de ML entrenados"""
    nombre = models.CharField(max_length=200)
    version = models.CharField(max_length=50)
    tipo_modelo = models.CharField(
        max_length=50,
        choices=[
            ('CLASIFICACION', 'Clasificación'),
            ('REGRESION', 'Regresión'),
            ('CLUSTERING', 'Clustering'),
            ('DETECCION_OBJETOS', 'Detección de Objetos'),
            ('RECONOCIMIENTO_PLACAS', 'Reconocimiento de Placas')
        ]
    )
    
    objetivo = models.TextField(help_text="Qué predice este modelo")
    
    # Métricas de rendimiento
    accuracy = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    precision = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    recall = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    f1_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    
    # Archivos del modelo
    archivo_modelo = models.FileField(upload_to='modelos/', null=True, blank=True)
    archivo_pesos = models.FileField(upload_to='modelos/pesos/', null=True, blank=True)
    
    fecha_entrenamiento = models.DateTimeField(auto_now_add=True)
    dataset_size = models.IntegerField(help_text="Cantidad de datos de entrenamiento")
    
    activo = models.BooleanField(default=False)
    notas = models.TextField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Modelo de Entrenamiento"
        verbose_name_plural = "Modelos de Entrenamiento"
        ordering = ['-fecha_entrenamiento']
    
    def __str__(self):
        return f"{self.nombre} v{self.version} - {self.tipo_modelo}"


class DatasetEntrenamiento(models.Model):
    """Datasets para entrenar modelos"""
    nombre = models.CharField(max_length=200)
    descripcion = models.TextField()
    
    tipo_datos = models.CharField(
        max_length=50,
        choices=[
            ('IMAGENES', 'Imágenes'),
            ('VIDEOS', 'Videos'),
            ('INFRACCIONES', 'Datos de Infracciones'),
            ('MIXTO', 'Mixto')
        ]
    )
    
    cantidad_registros = models.IntegerField()
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    
    archivo_dataset = models.FileField(upload_to='datasets/', null=True, blank=True)
    
    etiquetado_completo = models.BooleanField(default=False)
    
    class Meta:
        verbose_name = "Dataset de Entrenamiento"
        verbose_name_plural = "Datasets de Entrenamiento"
    
    def __str__(self):
        return f"{self.nombre} ({self.cantidad_registros} registros)"

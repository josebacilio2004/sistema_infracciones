from django.db import models
from django.utils import timezone
from infracciones.models import Infraccion, Vehiculo, TipoInfraccion

class Multa(models.Model):
    """Modelo para generar multas automáticas basadas en infracciones"""
    
    infraccion = models.OneToOneField(Infraccion, on_delete=models.CASCADE, related_name='multa')
    vehiculo = models.ForeignKey(Vehiculo, on_delete=models.CASCADE, related_name='multas')
    
    monto_base = models.DecimalField(max_digits=10, decimal_places=2)
    monto_total = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Descuentos y aumentos
    descuento_pronto_pago = models.DecimalField(
        max_digits=5, decimal_places=2, default=0,
        help_text="Descuento porcentual por pago dentro de 5 días"
    )
    aumento_reincidencia = models.DecimalField(
        max_digits=5, decimal_places=2, default=0,
        help_text="Aumento porcentual por infracciones previas"
    )
    
    # Fechas
    fecha_generacion = models.DateTimeField(default=timezone.now)
    fecha_vencimiento = models.DateTimeField()
    fecha_pago = models.DateTimeField(null=True, blank=True)
    
    # Estado
    estado = models.CharField(
        max_length=20,
        choices=[
            ('GENERADA', 'Generada'),
            ('NOTIFICADA', 'Notificada'),
            ('PAGADA', 'Pagada'),
            ('VENCIDA', 'Vencida'),
            ('ANULADA', 'Anulada')
        ],
        default='GENERADA'
    )
    
    numero_multa = models.CharField(max_length=20, unique=True)
    
    class Meta:
        verbose_name = "Multa"
        verbose_name_plural = "Multas"
        ordering = ['-fecha_generacion']
    
    def __str__(self):
        return f"Multa {self.numero_multa} - {self.vehiculo.placa} - S/. {self.monto_total}"
    
    def calcular_monto_final(self):
        """Calcula monto final con descuentos y aumentos"""
        monto = self.monto_base
        monto -= (monto * self.descuento_pronto_pago / 100)
        monto += (monto * self.aumento_reincidencia / 100)
        return monto

class GeneradorMultas(models.Manager):
    """Manager para generar multas automáticamente"""
    
    def crear_multa_desde_infraccion(self, infraccion):
        """Crea una multa basada en una infracción detectada"""
        
        # Obtener monto base
        monto_base = infraccion.tipo_infraccion.monto_multa
        
        # Calcular aumentos por reincidencia
        infracciones_previas = Infraccion.objects.filter(
            vehiculo=infraccion.vehiculo,
            fecha_hora__gte=timezone.now() - timezone.timedelta(days=30)
        ).count()
        
        aumento_reincidencia = min(infracciones_previas * 10, 50)
        
        # Crear multa
        numero_multa = f"MUL-{infraccion.id:06d}-{timezone.now().year}"
        
        multa = self.create(
            infraccion=infraccion,
            vehiculo=infraccion.vehiculo,
            monto_base=monto_base,
            monto_total=monto_base * (1 + aumento_reincidencia / 100),
            aumento_reincidencia=aumento_reincidencia,
            fecha_vencimiento=timezone.now() + timezone.timedelta(days=30),
            numero_multa=numero_multa
        )
        
        return multa

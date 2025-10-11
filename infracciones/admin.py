from django.contrib import admin
from django.utils.html import format_html
from .models import (
    TipoInfraccion, Vehiculo, Infraccion, 
    PerfilConductor, PrediccionAccidente, EventoDeteccion
)

@admin.register(TipoInfraccion)
class TipoInfraccionAdmin(admin.ModelAdmin):
    list_display = ['codigo', 'nombre', 'gravedad', 'monto_multa', 'puntos_licencia', 'activo']
    list_filter = ['gravedad', 'activo']
    search_fields = ['codigo', 'nombre']
    ordering = ['gravedad', 'codigo']

@admin.register(Vehiculo)
class VehiculoAdmin(admin.ModelAdmin):
    list_display = ['placa', 'marca', 'modelo', 'tipo_vehiculo', 'reportado_robado', 'total_infracciones', 'infracciones_ultimos_30_dias']
    list_filter = ['tipo_vehiculo', 'reportado_robado']
    search_fields = ['placa', 'propietario_nombre', 'propietario_documento']
    readonly_fields = ['fecha_registro']

@admin.register(Infraccion)
class InfraccionAdmin(admin.ModelAdmin):
    list_display = ['vehiculo', 'tipo_infraccion', 'fecha_hora', 'ubicacion', 'estado', 'confianza_deteccion', 'imagen_preview']
    list_filter = ['estado', 'tipo_infraccion', 'fecha_hora', 'camara']
    search_fields = ['vehiculo__placa', 'ubicacion']
    readonly_fields = ['fecha_hora', 'imagen_preview_large']
    date_hierarchy = 'fecha_hora'
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('vehiculo', 'tipo_infraccion', 'camara', 'fecha_hora', 'ubicacion', 'latitud', 'longitud')
        }),
        ('Datos de Infracción', {
            'fields': ('velocidad_detectada', 'velocidad_maxima', 'tiempo_luz_roja')
        }),
        ('Evidencia', {
            'fields': ('imagen_principal', 'imagen_placa', 'video_evidencia', 'imagen_preview_large')
        }),
        ('IA y Detección', {
            'fields': ('confianza_deteccion', 'modelo_ia_version')
        }),
        ('Estado y Verificación', {
            'fields': ('estado', 'verificada_por', 'fecha_verificacion', 'notas_verificacion', 'fecha_notificacion', 'fecha_pago')
        }),
    )
    
    def imagen_preview(self, obj):
        if obj.imagen_principal:
            return format_html('<img src="{}" width="50" height="50" />', obj.imagen_principal.url)
        return "Sin imagen"
    imagen_preview.short_description = "Vista Previa"
    
    def imagen_preview_large(self, obj):
        if obj.imagen_principal:
            return format_html('<img src="{}" width="400" />', obj.imagen_principal.url)
        return "Sin imagen"
    imagen_preview_large.short_description = "Imagen Principal"

@admin.register(PerfilConductor)
class PerfilConductorAdmin(admin.ModelAdmin):
    list_display = ['vehiculo', 'nivel_riesgo', 'puntuacion_riesgo', 'total_infracciones', 'probabilidad_reincidencia', 'probabilidad_accidente']
    list_filter = ['nivel_riesgo']
    search_fields = ['vehiculo__placa']
    readonly_fields = ['ultima_actualizacion']

@admin.register(PrediccionAccidente)
class PrediccionAccidenteAdmin(admin.ModelAdmin):
    list_display = ['ubicacion', 'probabilidad', 'periodo_prediccion', 'fecha_prediccion', 'infracciones_historicas']
    list_filter = ['periodo_prediccion', 'fecha_prediccion']
    search_fields = ['ubicacion']
    ordering = ['-probabilidad', '-fecha_prediccion']

@admin.register(EventoDeteccion)
class EventoDeteccionAdmin(admin.ModelAdmin):
    list_display = ['camara', 'tipo_evento', 'timestamp']
    list_filter = ['tipo_evento', 'camara', 'timestamp']
    search_fields = ['camara__ubicacion']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'

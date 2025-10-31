from django.contrib import admin
from .models import ModeloEntrenamiento, DatasetEntrenamiento

@admin.register(ModeloEntrenamiento)
class ModeloEntrenamientoAdmin(admin.ModelAdmin):
    list_display = ['nombre', 'version', 'tipo_modelo', 'accuracy', 'activo', 'fecha_entrenamiento']
    list_filter = ['tipo_modelo', 'activo', 'fecha_entrenamiento']
    search_fields = ['nombre', 'objetivo']
    readonly_fields = ['fecha_entrenamiento']

@admin.register(DatasetEntrenamiento)
class DatasetEntrenamientoAdmin(admin.ModelAdmin):
    list_display = ['nombre', 'tipo_datos', 'cantidad_registros', 'etiquetado_completo', 'fecha_creacion']
    list_filter = ['tipo_datos', 'etiquetado_completo']
    search_fields = ['nombre', 'descripcion']
    readonly_fields = ['fecha_creacion']

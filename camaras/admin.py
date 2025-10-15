from django.contrib import admin
from .models import Camara

@admin.register(Camara)
class CamaraAdmin(admin.ModelAdmin):
    list_display = ['ubicacion', 'ip', 'activa', 'fecha_instalacion', 'ultima_conexion']
    list_filter = ['activa', 'fecha_instalacion']
    search_fields = ['ubicacion', 'ip', 'descripcion']
    readonly_fields = ['fecha_instalacion', 'ultima_conexion']
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('ubicacion', 'ip', 'descripcion')
        }),
        ('Estado', {
            'fields': ('activa', 'fecha_instalacion', 'ultima_conexion')
        }),
    )
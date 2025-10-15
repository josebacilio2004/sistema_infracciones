from django.shortcuts import render, get_object_or_404
from django.db.models import Count, Q
from datetime import datetime, timedelta
from .models import Infraccion, TipoInfraccion

def lista_infracciones(request):
    """Lista todas las infracciones"""
    infracciones = Infraccion.objects.select_related(
        'vehiculo', 'tipo_infraccion', 'camara'
    ).order_by('-fecha_hora')[:100]
    
    context = {
        'infracciones': infracciones,
        'total': infracciones.count()
    }
    
    return render(request, 'infracciones/lista.html', context)

def detalle_infraccion(request, pk):
    """Detalle de una infracción"""
    infraccion = get_object_or_404(
        Infraccion.objects.select_related('vehiculo', 'tipo_infraccion', 'camara'),
        pk=pk
    )
    
    context = {
        'infraccion': infraccion
    }
    
    return render(request, 'infracciones/detalle.html', context)

def estadisticas(request):
    """Estadísticas de infracciones"""
    hoy = datetime.now().date()
    hace_7_dias = hoy - timedelta(days=7)
    
    # Infracciones por tipo
    por_tipo = Infraccion.objects.values('tipo_infraccion__nombre').annotate(
        total=Count('id')
    ).order_by('-total')
    
    # Infracciones por día (última semana)
    por_dia = []
    for i in range(7):
        dia = hoy - timedelta(days=i)
        count = Infraccion.objects.filter(fecha_hora__date=dia).count()
        por_dia.append({'dia': dia, 'count': count})
    
    context = {
        'por_tipo': por_tipo,
        'por_dia': reversed(por_dia),
        'total_semana': Infraccion.objects.filter(fecha_hora__date__gte=hace_7_dias).count()
    }
    
    return render(request, 'infracciones/estadisticas.html', context)

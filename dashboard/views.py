from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.db.models import Count, Q
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta
from camaras.models import Camara
from infracciones.models import Infraccion
import json

def home(request):
    hoy = datetime.now().date()
    
    total_camaras = Camara.objects.filter(activa=True).count()
    
    # Infracciones de hoy
    infracciones_hoy = Infraccion.objects.filter(
        fecha_hora__date=hoy
    ).count()
    
    # Alertas activas (infracciones no procesadas)
    alertas_activas = Infraccion.objects.filter(
        estado='DETECTADA'
    ).count()
    
    # Infracciones por tipo (últimas 24 horas)
    hace_24h = datetime.now() - timedelta(hours=24)
    infracciones_recientes = Infraccion.objects.filter(
        fecha_hora__gte=hace_24h
    ).values('tipo_infraccion__nombre').annotate(
        total=Count('id')
    )
    
    # Últimas infracciones para el feed de actividad
    ultimas_infracciones = Infraccion.objects.select_related(
        'vehiculo', 'tipo_infraccion', 'camara'
    ).order_by('-fecha_hora')[:10]
    
    # Estadísticas por hora (últimas 24 horas)
    infracciones_por_hora = []
    for i in range(24):
        hora_inicio = datetime.now() - timedelta(hours=24-i)
        hora_fin = hora_inicio + timedelta(hours=1)
        count = Infraccion.objects.filter(
            fecha_hora__gte=hora_inicio,
            fecha_hora__lt=hora_fin
        ).count()
        infracciones_por_hora.append({
            'hora': hora_inicio.strftime('%H:00'),
            'count': count
        })
    
    context = {
        'total_camaras': total_camaras,
        'infracciones_hoy': infracciones_hoy,
        'alertas_activas': alertas_activas,
        'infracciones_recientes': infracciones_recientes,
        'ultimas_infracciones': ultimas_infracciones,
        'infracciones_por_hora': infracciones_por_hora,
    }
    
    return render(request, "dashboard/home.html", context)

@csrf_exempt
def api_detecciones(request):
    """API para recibir detecciones desde el frontend"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Aquí procesarías la detección y la guardarías en la BD
            return JsonResponse({'status': 'success', 'message': 'Detección registrada'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    # GET: Devolver últimas detecciones
    detecciones = Infraccion.objects.select_related(
        'tipo_infraccion', 'vehiculo', 'camara'
    ).order_by('-fecha_hora')[:10].values(
        'id', 'tipo_infraccion__nombre', 'vehiculo__placa', 
        'velocidad_detectada', 'fecha_hora', 'camara__ubicacion'
    )
    
    return JsonResponse({'detecciones': list(detecciones)})

def video_feed(request):
    """Endpoint para streaming de video (opcional, para integración futura)"""
    # Este endpoint se puede usar para streaming desde el servidor
    # Por ahora, el video se maneja en el cliente
    return JsonResponse({'message': 'Video feed endpoint'})
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Count, Q
from datetime import datetime, timedelta
import json
import base64
import cv2
import numpy as np
from infracciones.models import Infraccion, Vehiculo, PerfilConductor
from infracciones.models_multa import Multa, GeneradorMultas
import logging

logger = logging.getLogger(__name__)

@csrf_exempt
def procesar_frame_deteccion(request):
    """Procesa frame con detector Tesseract - OPTIMIZADO PARA SERVIDOR"""
    try:
        data = json.loads(request.body)
        image_data = data.get('frame', '')
        
        if not image_data:
            return JsonResponse({'status': 'error', 'message': 'Frame vacío'}, status=400)
        
        # Decodificar frame
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JsonResponse({'status': 'error', 'message': 'Frame inválido'}, status=400)
        
        # Obtener detector
        from vision_ai.detector_tesseract_optimizado import DetectorTesseractOptimizado
        detector = DetectorTesseractOptimizado()
        
        # Procesar
        frame_procesado, detecciones = detector.procesar_frame(frame)
        
        # Convertir a base64
        _, buffer = cv2.imencode('.jpg', frame_procesado, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Obtener infracciones recientes
        infracciones_recientes = Infraccion.objects.filter(
            estado='DETECTADA'
        ).order_by('-fecha_hora')[:5].values(
            'vehiculo__placa', 'tipo_infraccion__nombre', 'velocidad_detectada'
        )
        
        return JsonResponse({
            'status': 'success',
            'frame': f'data:image/jpeg;base64,{frame_b64}',
            'detecciones': detecciones,
            'placas': [d['placa'] for d in detector.placas_detectadas[-5:]],
            'infracciones': list(infracciones_recientes)
        })
        
    except Exception as e:
        logger.error(f"Error procesando frame: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def dashboard_mejorado(request):
    """Dashboard mejorado con todas las métricas integradas"""
    
    hoy = datetime.now().date()
    hace_24h = datetime.now() - timedelta(hours=24)
    
    # Estadísticas generales
    infracciones_hoy = Infraccion.objects.filter(fecha_hora__date=hoy).count()
    multas_pendientes = Multa.objects.filter(estado='GENERADA').count()
    
    # Infracciones por tipo
    infracciones_por_tipo = Infraccion.objects.filter(
        fecha_hora__gte=hace_24h
    ).values('tipo_infraccion__nombre').annotate(count=Count('id')).order_by('-count')
    
    # Conductores de riesgo
    conductores_criticos = PerfilConductor.objects.filter(
        nivel_riesgo='CRITICO'
    ).select_related('vehiculo').order_by('-puntuacion_riesgo')[:5]
    
    # Últimas infracciones
    ultimas_infracciones = Infraccion.objects.select_related(
        'vehiculo', 'tipo_infraccion'
    ).order_by('-fecha_hora')[:10]
    
    context = {
        'infracciones_hoy': infracciones_hoy,
        'multas_pendientes': multas_pendientes,
        'infracciones_por_tipo': json.dumps(list(infracciones_por_tipo)),
        'conductores_criticos': conductores_criticos,
        'ultimas_infracciones': ultimas_infracciones,
    }
    
    return render(request, 'dashboard/home_mejorado.html', context)

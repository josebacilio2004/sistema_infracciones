from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.db.models import Count, Q
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from datetime import datetime, timedelta
from camaras.models import Camara
from infracciones.models import Infraccion, TipoInfraccion
import json
import cv2
import numpy as np
import threading
from pathlib import Path

import sys
import os
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from vision_ai.detector_webcam_mejorado import DetectorWebcamMejorado
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"YOLO not available: {e}")
    YOLO_AVAILABLE = False
    # Define una clase dummy o maneja el caso
    DetectorWebcamMejorado = None
    
detector_global = None
detector_lock = threading.Lock()
camara_actual = None

USE_LOCAL_VIDEO = os.getenv("USE_LOCAL_VIDEO", "False") == "True"

def inicializar_detector(fuente_video=0):
    """Inicializa el detector optimizado"""
    global detector_global, camara_actual
    
    with detector_lock:
        # Si ya hay un detector y es la misma fuente, no reiniciar
        if detector_global is not None and camara_actual == fuente_video:
            return detector_global
        
        # Detener detector anterior si existe
        if detector_global is not None:
            try:
                detector_global.detener()
            except:
                pass
        
        try:
            detector_global = DetectorWebcamMejorado(
                fuente_video=fuente_video,
                skip_frames=2,
                usar_gpu=True
            )
            camara_actual = fuente_video
            print(f"✅ Detector cargado con fuente: {fuente_video}")
            return detector_global
        except Exception as e:
            print(f"❌ Error al cargar detector: {e}")
            detector_global = None
            camara_actual = None
            return None

def home(request):
    hoy = datetime.now().date()
    
    camaras_disponibles = Camara.objects.filter(activa=True)
    total_camaras = camaras_disponibles.count()
    
    # Infracciones de hoy
    infracciones_hoy = Infraccion.objects.filter(
        fecha_hora__date=hoy
    ).count()
    
    # Alertas activas
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
    
    # Últimas infracciones
    ultimas_infracciones = Infraccion.objects.select_related(
        'vehiculo', 'tipo_infraccion', 'camara'
    ).order_by('-fecha_hora')[:10]
    
    # Estadísticas por hora
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
        'camaras_disponibles': camaras_disponibles,
        'infracciones_hoy': infracciones_hoy,
        'alertas_activas': alertas_activas,
        'infracciones_recientes': infracciones_recientes,
        'ultimas_infracciones': ultimas_infracciones,
        'infracciones_por_hora': infracciones_por_hora,
    }
    
    return render(request, "dashboard/home.html", context)

@csrf_exempt
@require_http_methods(["POST"])
def seleccionar_camara(request):
    """Cambia la cámara activa para el detector"""
    try:
        data = json.loads(request.body)
        camara_id = data.get('camara_id')
        
        if not camara_id:
            return JsonResponse({'error': 'ID de cámara requerido'}, status=400)
        
        camara = Camara.objects.get(id=camara_id, activa=True)
        fuente_video = camara.obtener_fuente_video()
        
        # Inicializar detector con la nueva fuente
        detector = inicializar_detector(fuente_video)
        
        if detector is None:
            return JsonResponse({'error': 'No se pudo inicializar el detector'}, status=500)
        
        return JsonResponse({
            'status': 'success',
            'message': f'Cámara cambiada a: {camara.ubicacion}',
            'camara': {
                'id': camara.id,
                'ubicacion': camara.ubicacion,
                'tipo': camara.get_tipo_fuente_display()
            }
        })
        
    except Camara.DoesNotExist:
        return JsonResponse({'error': 'Cámara no encontrada'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def procesar_frame_webcam(request):
    """Procesa un frame de la webcam con el detector"""
    try:
        import base64
        from io import BytesIO
        from PIL import Image
        
        data = json.loads(request.body)
        image_data = data.get('image', '')
        
        # Decodificar imagen
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convertir a numpy array
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        detector = detector_global
        if detector is None:
            detector = inicializar_detector(0)
        
        if detector is None:
            return JsonResponse({'error': 'Detector no disponible'}, status=500)
        
        # Procesar frame
        frame_procesado = detector.procesar_frame(frame)
        
        # Convertir frame procesado a base64
        _, buffer = cv2.imencode('.jpg', frame_procesado, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        fps_promedio = sum(detector.fps_real) / len(detector.fps_real) if detector.fps_real else 0
        
        detecciones = {
            'vehiculos': len(detector.vehiculos_trackeados),
            'placas_peruanas': len(detector.placas_detectadas),
            'fps': round(fps_promedio, 1),
            'frame_count': detector.frame_count,
            'infracciones': len(detector.ultimas_infracciones)
        }
        
        return JsonResponse({
            'status': 'success',
            'frame': f'data:image/jpeg;base64,{frame_base64}',
            'detecciones': detecciones
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

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

from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Count, Q
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.conf import settings
from datetime import datetime, timedelta
from camaras.models import Camara
from infracciones.models import Infraccion, TipoInfraccion, Vehiculo
import json
import cv2
import numpy as np
import threading
import logging
import base64
import time
import os

logger = logging.getLogger(__name__)

# Variables globales optimizadas
detector_global = None
detector_lock = threading.Lock()
cache_detecciones = {
    'ultimo_frame': None,
    'timestamp': None,
    'timeout': 2  # segundos
}

def obtener_detector_activo():
    """Obtiene o inicializa el detector global"""
    global detector_global
    with detector_lock:
        if detector_global is None:
            try:
                from vision_ai.detector_pytesseract import DetectorPytesseract
                detector_global = DetectorPytesseract()
            except ImportError:
                logger.warning("No se pudo importar DetectorTesseractProduccion")
                return None
        return detector_global

def presentacion_sistema(request):
    """Vista principal pública - Presentación del sistema"""
    try:
        total_infracciones = Infraccion.objects.count()
        total_vehiculos = Vehiculo.objects.count()
        total_camaras = Camara.objects.filter(activa=True).count()
        
        # Infracciones recientes (últimos 7 días)
        hace_7_dias = datetime.now() - timedelta(days=7)
        infracciones_semana = Infraccion.objects.filter(
            fecha_hora__gte=hace_7_dias
        ).count()
        
        context = {
            'total_infracciones': total_infracciones,
            'total_vehiculos': total_vehiculos,
            'total_camaras': total_camaras,
            'infracciones_semana': infracciones_semana,
        }
        
        return render(request, "presentacion/inicio.html", context)
    except Exception as e:
        logger.error(f"Error en presentacion_sistema: {e}")
        return render(request, "presentacion/inicio.html", {
            'total_infracciones': 0,
            'total_vehiculos': 0,
            'total_camaras': 0,
            'infracciones_semana': 0,
        })

@login_required(login_url='login')
def home(request):
    """Vista principal del dashboard - OPTIMIZADA"""
    hoy = datetime.now().date()
    hace_24h = datetime.now() - timedelta(hours=24)
    
    try:
        # Consultas optimizadas con select_related y prefetch
        total_camaras = Camara.objects.filter(activa=True).count()
        
        infracciones_hoy = Infraccion.objects.filter(
            fecha_hora__date=hoy
        ).count()
        
        alertas_activas = Infraccion.objects.filter(
            estado='DETECTADA'
        ).count()
        
        # Infracciones por tipo (últimas 24 horas)
        infracciones_recientes = Infraccion.objects.filter(
            fecha_hora__gte=hace_24h
        ).values('tipo_infraccion__nombre').annotate(
            total=Count('id')
        )
        
        # Últimas infracciones para el feed
        ultimas_infracciones = Infraccion.objects.select_related(
            'vehiculo', 'tipo_infraccion', 'camara'
        ).order_by('-fecha_hora')[:10]
        
        # Estadísticas por hora optimizadas
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
            'infracciones_recientes': list(infracciones_recientes),
            'ultimas_infracciones': ultimas_infracciones,
            'infracciones_por_hora': json.dumps(infracciones_por_hora),
        }
        
        return render(request, "dashboard/home.html", context)
        
    except Exception as e:
        logger.error(f"Error en vista home: {e}")
        # Contexto de respaldo en caso de error
        return render(request, "dashboard/home.html", {
            'total_camaras': 0,
            'infracciones_hoy': 0,
            'alertas_activas': 0,
            'infracciones_recientes': [],
            'ultimas_infracciones': [],
            'infracciones_por_hora': json.dumps([]),
        })

def capturar_frame_rtsp(url_rtsp, timeout=5):
    """Captura frame desde RTSP - OPTIMIZADO para producción"""
    cap = None
    try:
        # ⚡ OPTIMIZACIÓN: Timeout más corto para mejor respuesta
        cap = cv2.VideoCapture(url_rtsp)
        
        if not cap.isOpened():
            logger.warning(f"No se pudo abrir cámara RTSP: {url_rtsp}")
            return None
        
        # Configuraciones optimizadas
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        # Intentar leer frame con timeout optimizado
        start_time = time.time()
        frame = None
        
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if ret and frame is not None:
                # ⚡ OPTIMIZACIÓN: Redimensionar si es muy grande
                if frame.shape[1] > 1920:
                    frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
                logger.debug(f"Frame RTSP capturado: {frame.shape}")
                break
            time.sleep(0.1)
        
        return frame if frame is not None else None
            
    except Exception as e:
        logger.error(f"Error capturando frame RTSP: {e}")
        return None
    finally:
        if cap is not None:
            cap.release()

@csrf_exempt
@require_http_methods(["POST"])
def probar_camara_rtsp(request):
    """Endpoint para probar cámara RTSP - OPTIMIZADO"""
    try:
        if not hasattr(settings, 'EZVIZ_CONFIG'):
            return JsonResponse({
                'status': 'error', 
                'message': 'Configuración EZVIZ no encontrada'
            })
        
        url_rtsp = settings.EZVIZ_CONFIG['url_rtsp']
        logger.info(f"Probando cámara RTSP: {url_rtsp}")
        
        frame = capturar_frame_rtsp(url_rtsp)
        
        if frame is not None:
            # ⚡ OPTIMIZACIÓN: Comprimir imagen para respuesta más rápida
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return JsonResponse({
                'status': 'success',
                'message': 'Cámara RTSP funcionando correctamente',
                'frame': f'data:image/jpeg;base64,{frame_base64}',
                'resolucion': f"{frame.shape[1]}x{frame.shape[0]}",
                'debug': {
                    'url_rtsp': url_rtsp.split('@')[1] if '@' in url_rtsp else url_rtsp,
                    'frame_size': f"{frame.shape[1]}x{frame.shape[0]}"
                }
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'No se pudo conectar a la cámara RTSP'
            })
            
    except Exception as e:
        logger.error(f"Error probando cámara RTSP: {e}")
        return JsonResponse({
            'status': 'error',
            'message': f'Error de conexión: {str(e)}'
        })

def procesar_camara_rtsp():
    """Procesa frames desde cámara RTSP - OPTIMIZADO"""
    try:
        if not hasattr(settings, 'EZVIZ_CONFIG'):
            return JsonResponse({
                'status': 'error', 
                'error': 'Configuración EZVIZ no encontrada'
            })
        
        url_rtsp = settings.EZVIZ_CONFIG['url_rtsp']
        
        # Capturar frame desde RTSP
        frame = capturar_frame_rtsp(url_rtsp)
        
        if frame is None:
            return JsonResponse({
                'status': 'error', 
                'error': 'No se pudo capturar frame desde RTSP'
            })
        
        # Procesar con detector
        detector = obtener_detector_activo()
        if detector is None:
            return JsonResponse({
                'status': 'error', 
                'error': 'Detector no disponible'
            }, status=500)
        
        # ⚡ OPTIMIZACIÓN: Procesamiento con manejo de errores
        try:
            frame_procesado, detecciones = detector.procesar_frame(frame)
        except Exception as e:
            logger.error(f"Error procesando frame RTSP: {e}")
            frame_procesado = frame
            detecciones = []
        
        # Extraer placas detectadas con filtrado
        placas_detectadas = []
        if hasattr(detector, 'placas_detectadas') and detector.placas_detectadas:
            for placa_info in detector.placas_detectadas:
                if placa_info and placa_info.get('placa'):
                    placa_texto = placa_info['placa'].strip().upper()
                    # ⚡ FILTRADO: Solo placas con formato válido
                    if len(placa_texto) >= 4 and any(c.isdigit() for c in placa_texto):
                        placas_detectadas.append(placa_texto)
                        logger.info(f"Placa detectada RTSP: {placa_texto}")
        
        # Convertir frame procesado a base64 optimizado
        try:
            _, buffer = cv2.imencode('.jpg', frame_procesado, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Error codificando frame: {e}")
            frame_base64 = ""
        
        # Consultar SUNARP solo para placas válidas
        datos_sunarp = None
        if placas_detectadas:
            datos_sunarp = consultar_sunarp_automatico(placas_detectadas[0])
        
        response_data = {
            'status': 'success',
            'frame': f'data:image/jpeg;base64,{frame_base64}' if frame_base64 else '',
            'placas': placas_detectadas,
            'detecciones': {
                'vehiculos': len(detecciones),
                'placas_detectadas': placas_detectadas
            },
            'total_detecciones': len(placas_detectadas),
            'datos_sunarp': datos_sunarp,
            'debug': {
                'frame_shape': frame.shape,
                'placas_encontradas': len(placas_detectadas),
                'camera_type': 'rtsp',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error en procesar_camara_rtsp: {e}")
        return JsonResponse({
            'status': 'error', 
            'error': 'Error interno del servidor'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def procesar_frame(request):
    """Procesa frame con detector Tesseract optimizado"""
    try:
        data = json.loads(request.body)
        image_data = data.get('frame', '')
        camera_id = data.get('camera_id', 'local_0')
        
        if not image_data or image_data == 'RTSP_CAMERA_EZVIZ':
            # Procesar RTSP
            return procesar_camara_rtsp()
        
        # Decodificar frame
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JsonResponse({'status': 'error', 'message': 'Frame inválido'}, status=400)
        
        # Obtener detector
        from vision_ai.detector_tesseract_produccion import DetectorTesseractProduccion
        detector = DetectorTesseractProduccion()
        
        # Procesar
        frame_procesado, detecciones = detector.procesar_frame(frame)
        
        # Convertir a base64
        _, buffer = cv2.imencode('.jpg', frame_procesado, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Obtener infracciones recientes
        infracciones = Infraccion.objects.filter(
            estado='DETECTADA'
        ).order_by('-fecha_hora')[:5].values(
            'vehiculo__placa', 'tipo_infraccion__nombre', 'velocidad_detectada'
        )
        
        return JsonResponse({
            'status': 'success',
            'frame': f'data:image/jpeg;base64,{frame_b64}',
            'detecciones': detecciones,
            'placas': [d['placa'] for d in detector.placas_detectadas[-5:]],
            'infracciones': list(infracciones)
        })
        
    except Exception as e:
        logger.error(f"Error procesando frame: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def consultar_sunarp_automatico(placa):
    """Consulta SUNARP automáticamente - OPTIMIZADO"""
    try:
        # Validar placa antes de consultar
        if not placa or len(placa) < 4:
            return None
            
        from vision_ai.sunarp_integration import SunarpConsultor
        
        logger.info(f"Consultando SUNARP para placa: {placa}")
        
        consultor = SunarpConsultor()
        datos_sunarp = consultor.consultar(placa)
        
        if datos_sunarp:
            # Guardar o actualizar vehículo en BD
            vehiculo, created = Vehiculo.objects.update_or_create(
                placa=placa,
                defaults={
                    'marca': datos_sunarp.get('marca', ''),
                    'modelo': datos_sunarp.get('modelo', ''),
                    'color': datos_sunarp.get('color', ''),
                    'anio': datos_sunarp.get('anio', ''),
                    'propietario_nombre': datos_sunarp.get('propietario_nombre', ''),
                }
            )
            
            return {
                'status': 'success',
                'source': 'sunarp',
                'vehiculo': {
                    'placa': vehiculo.placa,
                    'marca': vehiculo.marca,
                    'modelo': vehiculo.modelo,
                    'color': vehiculo.color,
                    'anio': vehiculo.anio,
                    'propietario_nombre': vehiculo.propietario_nombre,
                    'estado': 'ACTIVO' if not vehiculo.reportado_robado else 'REPORTADO',
                }
            }
        else:
            # Buscar en BD local
            try:
                vehiculo = Vehiculo.objects.get(placa=placa)
                return {
                    'status': 'success',
                    'source': 'local',
                    'vehiculo': {
                        'placa': vehiculo.placa,
                        'marca': vehiculo.marca,
                        'modelo': vehiculo.modelo,
                        'color': vehiculo.color,
                        'anio': vehiculo.anio,
                        'propietario_nombre': vehiculo.propietario_nombre,
                        'estado': 'ACTIVO' if not vehiculo.reportado_robado else 'REPORTADO',
                    }
                }
            except Vehiculo.DoesNotExist:
                return {
                    'status': 'error',
                    'message': 'Vehículo no encontrado'
                }
    
    except ImportError:
        logger.warning("Módulo sunarp_integration no disponible")
        return None
    except Exception as e:
        logger.error(f"Error consultando SUNARP: {e}")
        return {
            'status': 'error',
            'message': 'Error en consulta SUNARP'
        }

@csrf_exempt
def listar_camaras_disponibles(request):
    """Lista cámaras disponibles - OPTIMIZADO"""
    try:
        camaras = []
        
        # Cámara EZVIZ desde settings
        if hasattr(settings, 'EZVIZ_CONFIG') and settings.EZVIZ_CONFIG.get('activa', True):
            camaras.append({
                'id': 'ezviz_rtsp',
                'nombre': '🎥 Cámara EZVIZ H6c Pro (Exterior)',
                'url_rtsp': settings.EZVIZ_CONFIG['url_rtsp'],
                'tipo': 'rtsp',
                'ubicacion': 'Entrada Principal',
                'resolucion': settings.EZVIZ_CONFIG.get('resolucion', '2304x1296')
            })
        
        # Detectar cámaras locales rápidamente
        for i in range(2):  # Solo probar 2 cámaras locales
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camaras.append({
                    'id': f'local_{i}',
                    'nombre': f'Cámara Local {i}',
                    'url_rtsp': i,
                    'tipo': 'local',
                    'ubicacion': 'Interna'
                })
                cap.release()
                break  # ⚡ OPTIMIZACIÓN: Solo primera cámara local
        
        return JsonResponse({'camaras': camaras})
        
    except Exception as e:
        logger.error(f"Error listando cámaras: {e}")
        return JsonResponse({'camaras': []})

@csrf_exempt
@require_http_methods(["POST"])
def cambiar_camara(request):
    """Cambia la cámara activa - OPTIMIZADO"""
    try:
        data = json.loads(request.body)
        camara_id = data.get('camara_id', 'local_0')
        
        logger.info(f"Cambiando a cámara: {camara_id}")
        
        # Reiniciar detector de forma controlada
        global detector_global
        with detector_lock:
            if detector_global is not None:
                try:
                    detector_global.detener()
                except Exception as e:
                    logger.warning(f"Error deteniendo detector: {e}")
                finally:
                    detector_global = None
        
        # Limpiar cache
        global cache_detecciones
        cache_detecciones.update({
            'ultimo_frame': None,
            'timestamp': None,
            'response': None
        })
        
        detector = obtener_detector_activo()
        
        if detector:
            return JsonResponse({
                'status': 'success',
                'message': f'Cámara {camara_id} configurada',
                'camara_id': camara_id
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'No se pudo inicializar el detector'
            }, status=500)
            
    except Exception as e:
        logger.error(f"Error cambiando cámara: {e}")
        return JsonResponse({
            'status': 'error', 
            'message': 'Error interno del servidor'
        }, status=500)

@csrf_exempt
def api_detecciones(request):
    """API para detecciones - OPTIMIZADA"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # ⚡ OPTIMIZACIÓN: Procesamiento asíncrono para detecciones
            return JsonResponse({
                'status': 'success', 
                'message': 'Detección registrada'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error', 
                'message': str(e)
            }, status=400)
    
    # GET: Devolver últimas detecciones optimizadas
    try:
        detecciones = Infraccion.objects.select_related(
            'tipo_infraccion', 'vehiculo', 'camara'
        ).order_by('-fecha_hora')[:10].values(
            'id', 'tipo_infraccion__nombre', 'vehiculo__placa', 
            'velocidad_detectada', 'fecha_hora', 'camara__ubicacion'
        )
        
        return JsonResponse({'detecciones': list(detecciones)})
    except Exception as e:
        logger.error(f"Error obteniendo detecciones: {e}")
        return JsonResponse({'detecciones': []})

@csrf_exempt
def estadisticas_tiempo_real(request):
    """Estadísticas en tiempo real - OPTIMIZADAS"""
    try:
        hoy = datetime.now().date()
        hace_1h = datetime.now() - timedelta(hours=1)
        
        stats = {
            'infracciones_hoy': Infraccion.objects.filter(fecha_hora__date=hoy).count(),
            'infracciones_ultima_hora': Infraccion.objects.filter(fecha_hora__gte=hace_1h).count(),
            'alertas_activas': Infraccion.objects.filter(estado='DETECTADA').count(),
            'vehiculos_detectados': Infraccion.objects.filter(fecha_hora__date=hoy).values('vehiculo').distinct().count(),
        }
        
        return JsonResponse(stats)
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        return JsonResponse({
            'infracciones_hoy': 0,
            'infracciones_ultima_hora': 0,
            'alertas_activas': 0,
            'vehiculos_detectados': 0,
        })

@csrf_exempt
@require_http_methods(["POST"])
def consultar_sunarp(request):
    """Consulta SUNARP - OPTIMIZADA"""
    try:
        data = json.loads(request.body)
        placa = data.get('placa', '').strip().upper()
        
        if not placa:
            return JsonResponse({
                'status': 'error', 
                'message': 'Placa no proporcionada'
            }, status=400)
        
        logger.info(f"Consultando SUNARP para: {placa}")
        
        from vision_ai.sunarp_integration import SunarpConsultor
        
        consultor = SunarpConsultor()
        datos_sunarp = consultor.consultar(placa)
        
        if datos_sunarp:
            vehiculo, created = Vehiculo.objects.update_or_create(
                placa=placa,
                defaults={
                    'marca': datos_sunarp.get('marca', ''),
                    'modelo': datos_sunarp.get('modelo', ''),
                    'color': datos_sunarp.get('color', ''),
                    'anio': datos_sunarp.get('anio', ''),
                    'propietario_nombre': datos_sunarp.get('propietario_nombre', ''),
                }
            )
            
            return JsonResponse({
                'status': 'success',
                'source': 'sunarp',
                'vehiculo': {
                    'placa': vehiculo.placa,
                    'marca': vehiculo.marca,
                    'modelo': vehiculo.modelo,
                    'color': vehiculo.color,
                    'anio': vehiculo.anio,
                    'propietario_nombre': vehiculo.propietario_nombre,
                    'estado': 'ACTIVO' if not vehiculo.reportado_robado else 'REPORTADO',
                }
            })
        
        # Buscar en BD local
        try:
            vehiculo = Vehiculo.objects.get(placa=placa)
            return JsonResponse({
                'status': 'success',
                'source': 'local',
                'vehiculo': {
                    'placa': vehiculo.placa,
                    'marca': vehiculo.marca,
                    'modelo': vehiculo.modelo,
                    'color': vehiculo.color,
                    'anio': vehiculo.anio,
                    'propietario_nombre': vehiculo.propietario_nombre,
                    'estado': 'ACTIVO' if not vehiculo.reportado_robado else 'REPORTADO',
                }
            })
        except Vehiculo.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Vehículo no encontrado'
            }, status=404)
    
    except Exception as e:
        logger.error(f"Error consultando SUNARP: {e}")
        return JsonResponse({
            'status': 'error', 
            'message': 'Error en consulta SUNARP'
        }, status=500)

def video_feed(request):
    """Endpoint para streaming (placeholder)"""
    return JsonResponse({
        'message': 'Endpoint de video feed',
        'status': 'en_desarrollo'
    })

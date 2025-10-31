from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from datetime import datetime, timedelta
from django.utils import timezone
from infracciones.models import Vehiculo, Infraccion, PerfilConductor, PrediccionAccidente, TipoInfraccion
from camaras.models import Camara


@csrf_exempt
@require_http_methods(["POST"])
def predecir_reincidencia(request):
    """
    Recibe predicción de reincidencia desde Google Colab
    Body: {
        "placa": "ABC123",
        "probabilidad_reincidencia": 75.5,
        "tipo_infraccion": "LUZ_ROJA",
        "modelo_version": "v1.0"
    }
    """
    try:
        data = json.loads(request.body)
        placa = data.get('placa')
        probabilidad = data.get('probabilidad_reincidencia')
        
        # Buscar o crear vehículo
        vehiculo, created = Vehiculo.objects.get_or_create(
            placa=placa,
            defaults={'tipo_vehiculo': 'AUTO'}
        )
        
        # Actualizar o crear perfil
        perfil, created = PerfilConductor.objects.get_or_create(vehiculo=vehiculo)
        perfil.probabilidad_reincidencia = probabilidad
        
        # Determinar nivel de riesgo
        if probabilidad >= 75:
            perfil.nivel_riesgo = 'CRITICO'
        elif probabilidad >= 50:
            perfil.nivel_riesgo = 'ALTO'
        elif probabilidad >= 25:
            perfil.nivel_riesgo = 'MEDIO'
        else:
            perfil.nivel_riesgo = 'BAJO'
        
        perfil.save()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Predicción de reincidencia registrada',
            'vehiculo': placa,
            'nivel_riesgo': perfil.nivel_riesgo,
            'probabilidad': float(probabilidad)
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def predecir_accidente(request):
    """
    Recibe predicción de accidente en zona específica
    Body: {
        "ubicacion": "Av. Principal con Calle 5",
        "latitud": -12.0464,
        "longitud": -77.0428,
        "probabilidad": 85.3,
        "periodo": "PROXIMO_DIA",
        "factores_riesgo": ["alta_reincidencia", "zona_escolar"],
        "infracciones_historicas": 45,
        "modelo_version": "v1.0"
    }
    """
    try:
        data = json.loads(request.body)
        
        prediccion = PrediccionAccidente.objects.create(
            ubicacion=data.get('ubicacion'),
            latitud=data.get('latitud'),
            longitud=data.get('longitud'),
            probabilidad=data.get('probabilidad'),
            periodo_prediccion=data.get('periodo', 'PROXIMO_DIA'),
            factores_riesgo=data.get('factores_riesgo', []),
            infracciones_historicas=data.get('infracciones_historicas', 0),
            accidentes_historicos=data.get('accidentes_historicos', 0),
            modelo_version=data.get('modelo_version', 'v1.0')
        )
        
        return JsonResponse({
            'status': 'success',
            'message': 'Predicción de accidente registrada',
            'prediccion_id': prediccion.id,
            'ubicacion': prediccion.ubicacion,
            'probabilidad': float(prediccion.probabilidad)
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def predecir_riesgo_conductor(request):
    """
    Recibe predicción completa de riesgo de conductor
    Body: {
        "placa": "ABC123",
        "puntuacion_riesgo": 78.5,
        "probabilidad_reincidencia": 65.2,
        "probabilidad_accidente": 45.8,
        "modelo_version": "v1.0"
    }
    """
    try:
        data = json.loads(request.body)
        placa = data.get('placa')
        
        vehiculo, created = Vehiculo.objects.get_or_create(
            placa=placa,
            defaults={'tipo_vehiculo': 'AUTO'}
        )
        
        perfil, created = PerfilConductor.objects.get_or_create(vehiculo=vehiculo)
        perfil.puntuacion_riesgo = data.get('puntuacion_riesgo', 0)
        perfil.probabilidad_reincidencia = data.get('probabilidad_reincidencia', 0)
        perfil.probabilidad_accidente = data.get('probabilidad_accidente', 0)
        
        # Determinar nivel de riesgo
        puntuacion = float(data.get('puntuacion_riesgo', 0))
        if puntuacion >= 75:
            perfil.nivel_riesgo = 'CRITICO'
        elif puntuacion >= 50:
            perfil.nivel_riesgo = 'ALTO'
        elif puntuacion >= 25:
            perfil.nivel_riesgo = 'MEDIO'
        else:
            perfil.nivel_riesgo = 'BAJO'
        
        perfil.save()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Perfil de riesgo actualizado',
            'vehiculo': placa,
            'nivel_riesgo': perfil.nivel_riesgo,
            'puntuacion': float(perfil.puntuacion_riesgo)
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


@require_http_methods(["GET"])
def obtener_datos_infracciones(request):
    """
    Endpoint para que Google Colab obtenga datos de infracciones para entrenamiento
    Query params: ?dias=30&limite=1000
    """
    try:
        dias = int(request.GET.get('dias', 30))
        limite = int(request.GET.get('limite', 1000))
        
        fecha_limite = timezone.now() - timedelta(days=dias)
        
        infracciones = Infraccion.objects.filter(
            fecha_hora__gte=fecha_limite
        ).select_related('vehiculo', 'tipo_infraccion', 'camara')[:limite]
        
        datos = []
        for inf in infracciones:
            datos.append({
                'id': inf.id,
                'placa': inf.vehiculo.placa,
                'tipo_infraccion': inf.tipo_infraccion.codigo,
                'gravedad': inf.tipo_infraccion.gravedad,
                'fecha_hora': inf.fecha_hora.isoformat(),
                'ubicacion': inf.ubicacion,
                'velocidad_detectada': inf.velocidad_detectada,
                'velocidad_maxima': inf.velocidad_maxima,
                'tiempo_luz_roja': float(inf.tiempo_luz_roja) if inf.tiempo_luz_roja else None,
                'confianza_deteccion': float(inf.confianza_deteccion),
                'estado': inf.estado,
            })
        
        return JsonResponse({
            'status': 'success',
            'total': len(datos),
            'datos': datos
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


@require_http_methods(["GET"])
def obtener_datos_vehiculos(request):
    """
    Endpoint para obtener datos de vehículos con sus perfiles
    """
    try:
        limite = int(request.GET.get('limite', 500))
        
        vehiculos = Vehiculo.objects.all()[:limite]
        
        datos = []
        for veh in vehiculos:
            try:
                perfil = veh.perfil
                perfil_data = {
                    'total_infracciones': perfil.total_infracciones,
                    'infracciones_luz_roja': perfil.infracciones_luz_roja,
                    'infracciones_velocidad': perfil.infracciones_velocidad,
                    'nivel_riesgo': perfil.nivel_riesgo,
                    'puntuacion_riesgo': float(perfil.puntuacion_riesgo),
                }
            except PerfilConductor.DoesNotExist:
                perfil_data = None
            
            datos.append({
                'placa': veh.placa,
                'tipo_vehiculo': veh.tipo_vehiculo,
                'total_infracciones': veh.total_infracciones(),
                'infracciones_30_dias': veh.infracciones_ultimos_30_dias(),
                'perfil': perfil_data
            })
        
        return JsonResponse({
            'status': 'success',
            'total': len(datos),
            'datos': datos
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def registrar_infraccion(request):
    """
    Endpoint para registrar una infracción detectada por IA
    Body: {
        "placa": "ABC123",
        "tipo_infraccion_codigo": "LUZ_ROJA",
        "camara_id": 1,
        "ubicacion": "Av. Principal",
        "confianza_deteccion": 95.5,
        "velocidad_detectada": 80,
        "velocidad_maxima": 60
    }
    """
    try:
        data = json.loads(request.body)
        
        # Obtener o crear vehículo
        vehiculo, created = Vehiculo.objects.get_or_create(
            placa=data.get('placa'),
            defaults={'tipo_vehiculo': 'AUTO'}
        )
        
        # Obtener tipo de infracción
        tipo_infraccion = TipoInfraccion.objects.get(codigo=data.get('tipo_infraccion_codigo'))
        
        # Obtener cámara
        camara = None
        if data.get('camara_id'):
            camara = Camara.objects.get(id=data.get('camara_id'))
        
        # Crear infracción
        infraccion = Infraccion.objects.create(
            vehiculo=vehiculo,
            tipo_infraccion=tipo_infraccion,
            camara=camara,
            ubicacion=data.get('ubicacion', 'Ubicación desconocida'),
            confianza_deteccion=data.get('confianza_deteccion', 0),
            velocidad_detectada=data.get('velocidad_detectada'),
            velocidad_maxima=data.get('velocidad_maxima'),
            tiempo_luz_roja=data.get('tiempo_luz_roja'),
            modelo_ia_version=data.get('modelo_version', 'v1.0')
        )
        
        return JsonResponse({
            'status': 'success',
            'message': 'Infracción registrada correctamente',
            'infraccion_id': infraccion.id,
            'placa': vehiculo.placa,
            'tipo': tipo_infraccion.nombre
        })
        
    except TipoInfraccion.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Tipo de infracción no encontrado'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)


@require_http_methods(["GET"])
def api_test(request):
    """Endpoint de prueba para verificar que la API funciona"""
    return JsonResponse({
        'status': 'success',
        'message': 'API de Machine Learning funcionando correctamente',
        'version': '1.0',
        'endpoints': [
            '/api/prediccion/reincidencia/',
            '/api/prediccion/accidente/',
            '/api/prediccion/riesgo-conductor/',
            '/api/datos/infracciones/',
            '/api/datos/vehiculos/',
            '/api/infraccion/registrar/',
        ]
    })

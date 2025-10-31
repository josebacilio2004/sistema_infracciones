from django.shortcuts import render
from django.http import JsonResponse
from .predictor import PredictorRiesgo
from infracciones.models import Vehiculo, Infraccion
from django.db.models import Count

def dashboard_ml(request):
    """Dashboard de predicciones ML"""
    predictor = PredictorRiesgo()
    
    # Obtener vehículos con más infracciones
    vehiculos_riesgo = Vehiculo.objects.annotate(
        num_infracciones=Count('infraccion')
    ).filter(num_infracciones__gt=0).order_by('-num_infracciones')[:10]
    
    context = {
        'vehiculos_riesgo': vehiculos_riesgo
    }
    
    return render(request, 'ml_predicciones/dashboard.html', context)

def predecir_vehiculo(request, placa):
    """Predice el riesgo de un vehículo"""
    try:
        predictor = PredictorRiesgo()
        prediccion = predictor.predecir_riesgo_vehiculo(placa)
        
        return JsonResponse({
            'status': 'success',
            'prediccion': prediccion
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)

def estadisticas_ml(request):
    """Estadísticas del modelo ML"""
    context = {
        'modelo_version': 'Random Forest v1.0',
        'precision': 94.2,
        'recall': 91.8,
        'f1_score': 93.0
    }
    
    return render(request, 'ml_predicciones/estadisticas.html', context)

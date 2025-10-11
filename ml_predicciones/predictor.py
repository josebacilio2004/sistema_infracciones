"""
Módulo de predicción ML integrado con Django
Predice riesgo de reincidencia y accidentes
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from django.utils import timezone

class PredictorRiesgo:
    """Predictor de riesgo usando modelos de Machine Learning"""
    
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.feature_names = None
        self.modelo_cargado = False
        
        # Intentar cargar modelo entrenado
        self.cargar_modelo()
    
    def cargar_modelo(self):
        """Carga el modelo ML entrenado desde el notebook"""
        try:
            base_dir = Path(__file__).resolve().parent.parent
            modelo_path = base_dir / 'notebooks' / 'modelo_reincidencia.pkl'
            scaler_path = base_dir / 'notebooks' / 'scaler.pkl'
            
            if modelo_path.exists() and scaler_path.exists():
                self.modelo = joblib.load(modelo_path)
                self.scaler = joblib.load(scaler_path)
                self.feature_names = [
                    'total_infracciones',
                    'infracciones_graves',
                    'infracciones_leves',
                    'velocidad_promedio',
                    'tasa_infracciones_mes',
                    'hora_promedio'
                ]
                self.modelo_cargado = True
                print("✅ Modelo ML cargado correctamente")
            else:
                print("⚠️  Modelo ML no encontrado. Ejecuta el notebook primero.")
                self.modelo_cargado = False
                
        except Exception as e:
            print(f"⚠️  Error al cargar modelo ML: {e}")
            self.modelo_cargado = False
    
    def calcular_features_vehiculo(self, placa):
        """Calcula features de un vehículo para predicción"""
        from infracciones.models import Vehiculo, Infraccion
        
        try:
            vehiculo = Vehiculo.objects.get(placa=placa)
            infracciones = Infraccion.objects.filter(vehiculo=vehiculo)
            
            if infracciones.count() == 0:
                # Vehículo sin historial
                return {
                    'total_infracciones': 0,
                    'infracciones_graves': 0,
                    'infracciones_leves': 0,
                    'velocidad_promedio': 50.0,
                    'tasa_infracciones_mes': 0.0,
                    'hora_promedio': 12.0
                }
            
            # Calcular estadísticas
            total = infracciones.count()
            
            # Contar graves (GRAVE y MUY_GRAVE)
            graves = infracciones.filter(
                tipo_infraccion__gravedad__in=['GRAVE', 'MUY_GRAVE']
            ).count()
            leves = total - graves
            
            # Velocidad promedio (solo de infracciones con velocidad)
            velocidades = infracciones.filter(
                velocidad_detectada__isnull=False
            ).values_list('velocidad_detectada', flat=True)
            velocidad_prom = np.mean(list(velocidades)) if velocidades else 50.0
            
            # Tasa de infracciones por mes
            fecha_primera = infracciones.order_by('fecha_hora').first().fecha_hora
            fecha_ultima = infracciones.order_by('-fecha_hora').first().fecha_hora
            dias_diferencia = (fecha_ultima - fecha_primera).days
            
            if dias_diferencia == 0:
                tasa_mes = total
            else:
                tasa_mes = (total / dias_diferencia) * 30
            
            # Hora promedio
            horas = [inf.fecha_hora.hour for inf in infracciones]
            hora_prom = np.mean(horas) if horas else 12.0
            
            return {
                'total_infracciones': total,
                'infracciones_graves': graves,
                'infracciones_leves': leves,
                'velocidad_promedio': float(velocidad_prom),
                'tasa_infracciones_mes': float(tasa_mes),
                'hora_promedio': float(hora_prom)
            }
            
        except Vehiculo.DoesNotExist:
            # Vehículo nuevo
            return {
                'total_infracciones': 0,
                'infracciones_graves': 0,
                'infracciones_leves': 0,
                'velocidad_promedio': 50.0,
                'tasa_infracciones_mes': 0.0,
                'hora_promedio': 12.0
            }
    
    def predecir_riesgo_vehiculo(self, placa):
        """Predice el riesgo de reincidencia de un vehículo"""
        from infracciones.models import PerfilConductor, Vehiculo
        
        # Calcular features
        features = self.calcular_features_vehiculo(placa)
        
        # Si no hay modelo, usar heurística simple
        if not self.modelo_cargado:
            return self._prediccion_heuristica(features)
        
        # Preparar datos para el modelo
        X = pd.DataFrame([features])[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Predecir
        probabilidad = self.modelo.predict_proba(X_scaled)[0][1] * 100
        es_reincidente = probabilidad > 50
        
        # Determinar nivel de riesgo
        if probabilidad < 25:
            nivel_riesgo = 'BAJO'
        elif probabilidad < 50:
            nivel_riesgo = 'MEDIO'
        elif probabilidad < 75:
            nivel_riesgo = 'ALTO'
        else:
            nivel_riesgo = 'CRITICO'
        
        # Actualizar perfil del conductor
        try:
            vehiculo = Vehiculo.objects.get(placa=placa)
            perfil, created = PerfilConductor.objects.get_or_create(vehiculo=vehiculo)
            
            perfil.total_infracciones = features['total_infracciones']
            perfil.infracciones_graves = features['infracciones_graves']
            perfil.puntuacion_riesgo = probabilidad
            perfil.nivel_riesgo = nivel_riesgo
            perfil.probabilidad_reincidencia = probabilidad
            perfil.save()
            
        except Exception as e:
            print(f"⚠️  Error al actualizar perfil: {e}")
        
        return {
            'placa': placa,
            'es_reincidente': es_reincidente,
            'probabilidad_reincidencia': probabilidad,
            'nivel_riesgo': nivel_riesgo,
            'features': features
        }
    
    def _prediccion_heuristica(self, features):
        """Predicción simple sin modelo ML (fallback)"""
        total = features['total_infracciones']
        graves = features['infracciones_graves']
        tasa = features['tasa_infracciones_mes']
        
        # Heurística simple
        puntuacion = (total * 10) + (graves * 20) + (tasa * 5)
        probabilidad = min(puntuacion, 100)
        
        if probabilidad < 25:
            nivel_riesgo = 'BAJO'
        elif probabilidad < 50:
            nivel_riesgo = 'MEDIO'
        elif probabilidad < 75:
            nivel_riesgo = 'ALTO'
        else:
            nivel_riesgo = 'CRITICO'
        
        return {
            'placa': 'DESCONOCIDA',
            'es_reincidente': probabilidad > 50,
            'probabilidad_reincidencia': probabilidad,
            'nivel_riesgo': nivel_riesgo,
            'features': features
        }
    
    def predecir_zona_riesgo(self, ubicacion, latitud, longitud):
        """Predice riesgo de accidente en una zona específica"""
        from infracciones.models import Infraccion, PrediccionAccidente
        
        # Contar infracciones históricas en la zona (radio de 1km aprox)
        infracciones_zona = Infraccion.objects.filter(
            ubicacion__icontains=ubicacion
        ).count()
        
        # Calcular probabilidad basada en historial
        if infracciones_zona == 0:
            probabilidad = 5.0
        elif infracciones_zona < 10:
            probabilidad = 20.0
        elif infracciones_zona < 50:
            probabilidad = 50.0
        else:
            probabilidad = 80.0
        
        # Crear predicción
        prediccion = PrediccionAccidente.objects.create(
            ubicacion=ubicacion,
            latitud=latitud,
            longitud=longitud,
            periodo_prediccion='PROXIMO_DIA',
            probabilidad=probabilidad,
            factores_riesgo={
                'infracciones_historicas': infracciones_zona,
                'tipo_zona': 'urbana'
            },
            infracciones_historicas=infracciones_zona,
            modelo_version='heuristic_v1.0'
        )
        
        return prediccion

"""
Script para entrenar el modelo ML desde el notebook
Ejecutar: python scripts/entrenar_modelo_ml.py
"""
import os
import sys
import django
from pathlib import Path

# Configurar Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from infracciones.models import Infraccion, Vehiculo

def generar_dataset_desde_bd():
    """Genera dataset de entrenamiento desde la base de datos"""
    print("📊 Generando dataset desde base de datos...")
    
    vehiculos = Vehiculo.objects.all()
    
    if vehiculos.count() == 0:
        print("⚠️  No hay datos en la base de datos. Generando datos sintéticos...")
        return generar_dataset_sintetico()
    
    features = []
    
    for vehiculo in vehiculos:
        infracciones = Infraccion.objects.filter(vehiculo=vehiculo)
        
        if infracciones.count() == 0:
            continue
        
        total = infracciones.count()
        graves = infracciones.filter(
            tipo_infraccion__gravedad__in=['GRAVE', 'MUY_GRAVE']
        ).count()
        leves = total - graves
        
        velocidades = infracciones.filter(
            velocidad_detectada__isnull=False
        ).values_list('velocidad_detectada', flat=True)
        velocidad_prom = np.mean(list(velocidades)) if velocidades else 50.0
        
        fecha_primera = infracciones.order_by('fecha_hora').first().fecha_hora
        fecha_ultima = infracciones.order_by('-fecha_hora').first().fecha_hora
        dias_diferencia = (fecha_ultima - fecha_primera).days
        
        if dias_diferencia == 0:
            tasa_mes = total
        else:
            tasa_mes = (total / dias_diferencia) * 30
        
        horas = [inf.fecha_hora.hour for inf in infracciones]
        hora_prom = np.mean(horas) if horas else 12.0
        
        # Etiqueta: es reincidente si tiene más de 3 infracciones
        es_reincidente = 1 if total > 3 else 0
        
        features.append([
            total, graves, leves, velocidad_prom, tasa_mes, hora_prom, es_reincidente
        ])
    
    if len(features) < 10:
        print("⚠️  Pocos datos reales. Complementando con datos sintéticos...")
        return generar_dataset_sintetico()
    
    df = pd.DataFrame(features, columns=[
        'total_infracciones', 'infracciones_graves', 'infracciones_leves',
        'velocidad_promedio', 'tasa_infracciones_mes', 'hora_promedio',
        'es_reincidente'
    ])
    
    print(f"✅ Dataset generado: {len(df)} registros")
    return df

def generar_dataset_sintetico(n_registros=500):
    """Genera dataset sintético para entrenamiento"""
    print(f"🎲 Generando {n_registros} registros sintéticos...")
    
    np.random.seed(42)
    
    data = []
    for _ in range(n_registros):
        total = np.random.randint(1, 20)
        graves = np.random.randint(0, total)
        leves = total - graves
        velocidad = np.random.normal(60, 15)
        tasa_mes = np.random.uniform(0.5, 10)
        hora = np.random.uniform(0, 24)
        
        # Etiqueta basada en heurística
        es_reincidente = 1 if (total > 5 or graves > 3 or tasa_mes > 5) else 0
        
        data.append([total, graves, leves, velocidad, tasa_mes, hora, es_reincidente])
    
    df = pd.DataFrame(data, columns=[
        'total_infracciones', 'infracciones_graves', 'infracciones_leves',
        'velocidad_promedio', 'tasa_infracciones_mes', 'hora_promedio',
        'es_reincidente'
    ])
    
    print(f"✅ Dataset sintético generado: {len(df)} registros")
    return df

def entrenar_modelo(df):
    """Entrena el modelo de clasificación"""
    print("\n🤖 Entrenando modelo de Machine Learning...")
    
    X = df[['total_infracciones', 'infracciones_graves', 'infracciones_leves',
            'velocidad_promedio', 'tasa_infracciones_mes', 'hora_promedio']]
    y = df['es_reincidente']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar Random Forest
    modelo = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    modelo.fit(X_train_scaled, y_train)
    
    # Evaluar
    y_pred = modelo.predict(X_test_scaled)
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n📈 Métricas del modelo:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   AUC-ROC: {auc:.2%}")
    print("\n" + classification_report(y_test, y_pred, 
                                       target_names=['No Reincidente', 'Reincidente']))
    
    return modelo, scaler

def guardar_modelo(modelo, scaler):
    """Guarda el modelo entrenado"""
    print("\n💾 Guardando modelo...")
    
    notebooks_dir = BASE_DIR / 'notebooks'
    notebooks_dir.mkdir(exist_ok=True)
    
    joblib.dump(modelo, notebooks_dir / 'modelo_reincidencia.pkl')
    joblib.dump(scaler, notebooks_dir / 'scaler.pkl')
    
    print("✅ Modelo guardado en notebooks/")

def main():
    print("=" * 60)
    print("🧠 ENTRENAMIENTO DE MODELO ML")
    print("📚 Sistema de Predicción de Reincidencia")
    print("=" * 60)
    print()
    
    # Generar dataset
    df = generar_dataset_desde_bd()
    
    # Entrenar
    modelo, scaler = entrenar_modelo(df)
    
    # Guardar
    guardar_modelo(modelo, scaler)
    
    print("\n✅ Proceso completado exitosamente")
    print("🚀 Ahora puedes ejecutar: python vision_ai/detector_webcam.py")

if __name__ == "__main__":
    main()

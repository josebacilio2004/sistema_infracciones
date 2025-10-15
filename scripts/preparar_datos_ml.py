"""
Script para preparar datos de entrenamiento ML
Extrae features de las infracciones existentes
"""
import os
import sys
import django
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Configurar Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

from infracciones.models import Infraccion, Vehiculo

def extraer_features():
    """Extrae features de los vehículos para entrenamiento ML"""
    print("Extrayendo features de vehículos...")
    
    vehiculos = Vehiculo.objects.all()
    datos = []
    
    for vehiculo in vehiculos:
        infracciones = Infraccion.objects.filter(vehiculo=vehiculo)
        
        if infracciones.count() == 0:
            continue
        
        # Calcular features
        total = infracciones.count()
        graves = infracciones.filter(tipo_infraccion__gravedad__in=['GRAVE', 'MUY_GRAVE']).count()
        leves = total - graves
        
        # Velocidad promedio
        velocidades = infracciones.filter(velocidad_detectada__isnull=False).values_list('velocidad_detectada', flat=True)
        velocidad_prom = np.mean(list(velocidades)) if velocidades else 50.0
        
        # Tasa de infracciones por mes
        fecha_primera = infracciones.order_by('fecha_hora').first().fecha_hora
        fecha_ultima = infracciones.order_by('-fecha_hora').first().fecha_hora
        dias_diferencia = (fecha_ultima - fecha_primera).days
        tasa_mes = (total / max(dias_diferencia, 1)) * 30
        
        # Hora promedio
        horas = [inf.fecha_hora.hour for inf in infracciones]
        hora_prom = np.mean(horas) if horas else 12.0
        
        # Etiqueta: es reincidente si tiene más de 3 infracciones
        es_reincidente = 1 if total > 3 else 0
        
        datos.append({
            'placa': vehiculo.placa,
            'total_infracciones': total,
            'infracciones_graves': graves,
            'infracciones_leves': leves,
            'velocidad_promedio': velocidad_prom,
            'tasa_infracciones_mes': tasa_mes,
            'hora_promedio': hora_prom,
            'es_reincidente': es_reincidente
        })
    
    df = pd.DataFrame(datos)
    
    # Guardar dataset
    output_path = BASE_DIR / 'notebooks' / 'dataset_infracciones.csv'
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset guardado: {output_path}")
    print(f"   Total de registros: {len(df)}")
    print(f"   Reincidentes: {df['es_reincidente'].sum()}")
    print(f"   No reincidentes: {len(df) - df['es_reincidente'].sum()}")
    
    return df

def generar_datos_sinteticos(n=1000):
    """Genera datos sintéticos para entrenamiento si no hay suficientes datos reales"""
    print(f"\nGenerando {n} registros sintéticos...")
    
    np.random.seed(42)
    
    datos = []
    for i in range(n):
        # Generar features aleatorias pero realistas
        total_inf = np.random.poisson(3)
        graves = np.random.binomial(total_inf, 0.3)
        leves = total_inf - graves
        velocidad = np.random.normal(70, 15)
        tasa = np.random.exponential(2)
        hora = np.random.normal(14, 4) % 24
        
        # Etiqueta basada en reglas
        puntuacion = (total_inf * 10) + (graves * 20) + (tasa * 5)
        es_reincidente = 1 if puntuacion > 50 else 0
        
        datos.append({
            'placa': f'SYN-{i:04d}',
            'total_infracciones': max(0, total_inf),
            'infracciones_graves': max(0, graves),
            'infracciones_leves': max(0, leves),
            'velocidad_promedio': max(30, min(150, velocidad)),
            'tasa_infracciones_mes': max(0, tasa),
            'hora_promedio': hora,
            'es_reincidente': es_reincidente
        })
    
    df = pd.DataFrame(datos)
    
    # Guardar dataset sintético
    output_path = BASE_DIR / 'notebooks' / 'dataset_sintetico.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset sintético guardado: {output_path}")
    print(f"   Reincidentes: {df['es_reincidente'].sum()}")
    print(f"   No reincidentes: {len(df) - df['es_reincidente'].sum()}")
    
    return df

def main():
    print("=" * 60)
    print("PREPARACIÓN DE DATOS PARA ML")
    print("=" * 60)
    print()
    
    try:
        # Extraer datos reales
        df_real = extraer_features()
        
        # Si no hay suficientes datos, generar sintéticos
        if len(df_real) < 100:
            print("\n⚠️  Pocos datos reales, generando datos sintéticos...")
            df_sintetico = generar_datos_sinteticos(1000)
            
            # Combinar datasets
            df_combinado = pd.concat([df_real, df_sintetico], ignore_index=True)
            output_path = BASE_DIR / 'notebooks' / 'dataset_completo.csv'
            df_combinado.to_csv(output_path, index=False)
            print(f"\n✅ Dataset completo guardado: {output_path}")
            print(f"   Total de registros: {len(df_combinado)}")
        
        print("\n" + "=" * 60)
        print("✅ PREPARACIÓN COMPLETADA")
        print("=" * 60)
        print("\nSiguiente paso: python scripts/entrenar_modelo_ml.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

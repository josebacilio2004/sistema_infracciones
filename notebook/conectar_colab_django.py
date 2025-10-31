"""
Script para conectar Google Colab con Django
Ejecuta este script desde Google Colab para enviar predicciones al sistema Django
"""

import requests
import json

# ============================================
# CONFIGURACIÓN
# ============================================

# Si estás ejecutando Django localmente, usa ngrok para exponer tu servidor
# Instala ngrok: https://ngrok.com/download
# Ejecuta: ngrok http 8000
# Copia la URL que te da ngrok (ejemplo: https://abc123.ngrok.io)

DJANGO_URL = "http://127.0.0.1:8000"  # Cambia esto por tu URL de ngrok si usas Colab
API_ENDPOINT = f"{DJANGO_URL}/api"

# ============================================
# FUNCIONES PARA ENVIAR PREDICCIONES
# ============================================

def test_conexion():
    """Prueba la conexión con el servidor Django"""
    url = f"{API_ENDPOINT}/test/"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ Conexión exitosa con Django")
            print(f"   Respuesta: {response.json()}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error de conexión: {str(e)}")
        print("   Verifica que Django esté corriendo y la URL sea correcta")
        return False


def enviar_prediccion_reincidencia(placa, probabilidad, tipo_infraccion, modelo_version="v1.0"):
    """
    Envía una predicción de reincidencia al sistema Django
    
    Args:
        placa: Placa del vehículo
        probabilidad: Probabilidad de reincidencia (0-100)
        tipo_infraccion: Tipo de infracción
        modelo_version: Versión del modelo usado
    """
    url = f"{API_ENDPOINT}/prediccion/reincidencia/"
    
    data = {
        "placa": placa,
        "probabilidad_reincidencia": probabilidad,
        "tipo_infraccion": tipo_infraccion,
        "modelo_version": modelo_version
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 201:
            resultado = response.json()
            print(f"✅ Predicción de reincidencia guardada para {placa}")
            print(f"   ID: {resultado.get('id')}")
            print(f"   Probabilidad: {probabilidad}%")
            return resultado
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error de conexión: {str(e)}")
        return None


def enviar_prediccion_accidente(ubicacion, latitud, longitud, probabilidad, 
                                periodo, factores_riesgo, infracciones_historicas):
    """
    Envía una predicción de accidente al sistema Django
    
    Args:
        ubicacion: Descripción de la ubicación
        latitud: Latitud de la zona
        longitud: Longitud de la zona
        probabilidad: Probabilidad de accidente (0-100)
        periodo: Periodo de predicción (PROXIMO_DIA, PROXIMA_SEMANA, etc.)
        factores_riesgo: Lista de factores de riesgo
        infracciones_historicas: Número de infracciones históricas en la zona
    """
    url = f"{API_ENDPOINT}/prediccion/accidente/"
    
    data = {
        "ubicacion": ubicacion,
        "latitud": latitud,
        "longitud": longitud,
        "probabilidad": probabilidad,
        "periodo": periodo,
        "factores_riesgo": factores_riesgo,
        "infracciones_historicas": infracciones_historicas
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 201:
            resultado = response.json()
            print(f"✅ Predicción de accidente guardada para {ubicacion}")
            print(f"   ID: {resultado.get('id')}")
            print(f"   Probabilidad: {probabilidad}%")
            return resultado
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error de conexión: {str(e)}")
        return None


def enviar_perfil_riesgo(placa, puntuacion_riesgo, prob_reincidencia, prob_accidente):
    """
    Envía un perfil de riesgo de conductor al sistema Django
    
    Args:
        placa: Placa del vehículo
        puntuacion_riesgo: Puntuación de riesgo (0-100)
        prob_reincidencia: Probabilidad de reincidencia (0-100)
        prob_accidente: Probabilidad de accidente (0-100)
    """
    url = f"{API_ENDPOINT}/prediccion/riesgo-conductor/"
    
    data = {
        "placa": placa,
        "puntuacion_riesgo": puntuacion_riesgo,
        "probabilidad_reincidencia": prob_reincidencia,
        "probabilidad_accidente": prob_accidente
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 201:
            resultado = response.json()
            print(f"✅ Perfil de riesgo actualizado para {placa}")
            print(f"   Puntuación: {puntuacion_riesgo}")
            print(f"   Nivel: {resultado.get('nivel_riesgo', 'N/A')}")
            return resultado
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error de conexión: {str(e)}")
        return None


# ============================================
# EJEMPLOS DE USO
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("CONECTANDO GOOGLE COLAB CON DJANGO - SISTEMA DE PREDICCIONES ML")
    print("=" * 70)
    
    # Probar conexión primero
    print("\n🔌 Probando conexión con Django...")
    print("-" * 70)
    if not test_conexion():
        print("\n⚠️  No se pudo conectar con Django. Verifica:")
        print("   1. Django está corriendo: python manage.py runserver")
        print("   2. La URL es correcta: " + DJANGO_URL)
        print("   3. Si usas Colab, necesitas ngrok para exponer tu servidor local")
        exit(1)
    
    # Ejemplo 1: Predicción de reincidencia
    print("\n📊 Ejemplo 1: Predicción de reincidencia en luz roja")
    print("-" * 70)
    enviar_prediccion_reincidencia(
        placa="ABC-123",
        probabilidad=75.5,
        tipo_infraccion="LUZ_ROJA",
        modelo_version="v1.0"
    )
    
    # Ejemplo 2: Predicción de accidente
    print("\n📊 Ejemplo 2: Predicción de accidente en zona de alto riesgo")
    print("-" * 70)
    enviar_prediccion_accidente(
        ubicacion="Av. Principal con Calle 5",
        latitud=-12.0464,
        longitud=-77.0428,
        probabilidad=85.3,
        periodo="PROXIMO_DIA",
        factores_riesgo=["alta_reincidencia", "zona_escolar", "hora_pico"],
        infracciones_historicas=45
    )
    
    # Ejemplo 3: Perfil de riesgo de conductor
    print("\n📊 Ejemplo 3: Actualizar perfil de riesgo de conductor")
    print("-" * 70)
    enviar_perfil_riesgo(
        placa="ABC-123",
        puntuacion_riesgo=78.5,
        prob_reincidencia=65.2,
        prob_accidente=45.8
    )
    
    print("\n" + "=" * 70)
    print("✅ Todas las pruebas completadas exitosamente")
    print("=" * 70)
    print("\n💡 Próximos pasos:")
    print("   1. Revisa los datos en: http://127.0.0.1:8000/admin/")
    print("   2. Entrena tus modelos con datos reales")
    print("   3. Integra con las cámaras de detección")

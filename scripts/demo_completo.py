"""
Script de demostración completa del sistema
Ejecuta todas las funcionalidades principales
"""
import os
import sys
import django
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

from infracciones.models import Infraccion, Vehiculo, TipoInfraccion, PerfilConductor
from camaras.models import Camara
from ml_predicciones.predictor import PredictorRiesgo
from django.utils import timezone
from datetime import timedelta
import random

def crear_datos_demo():
    """Crea datos de demostración para pruebas"""
    print("🎬 Creando datos de demostración...\n")
    
    # Obtener tipos de infracciones
    tipos = list(TipoInfraccion.objects.all())
    if not tipos:
        print("❌ No hay tipos de infracciones. Ejecuta: python scripts/inicializar_sistema.py")
        return
    
    # Obtener cámara
    camara = Camara.objects.first()
    if not camara:
        print("❌ No hay cámaras. Ejecuta: python scripts/inicializar_sistema.py")
        return
    
    # Crear vehículos de prueba
    placas_demo = [
        "ABC-123", "XYZ-789", "DEF-456", "GHI-321", "JKL-654",
        "MNO-987", "PQR-147", "STU-258", "VWX-369", "YZA-741"
    ]
    
    print("🚗 Creando vehículos de prueba...")
    vehiculos = []
    for placa in placas_demo:
        vehiculo, created = Vehiculo.objects.get_or_create(
            placa=placa,
            defaults={
                'tipo_vehiculo': random.choice(['AUTO', 'MOTO', 'CAMION']),
                'marca': random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet']),
                'color': random.choice(['Rojo', 'Azul', 'Negro', 'Blanco', 'Gris'])
            }
        )
        vehiculos.append(vehiculo)
        if created:
            print(f"  ✅ {placa}")
    
    # Crear infracciones de prueba
    print("\n⚠️  Creando infracciones de prueba...")
    infracciones_creadas = 0
    
    for vehiculo in vehiculos:
        # Cada vehículo tiene entre 1 y 8 infracciones
        num_infracciones = random.randint(1, 8)
        
        for i in range(num_infracciones):
            tipo = random.choice(tipos)
            
            # Fecha aleatoria en los últimos 90 días
            dias_atras = random.randint(0, 90)
            fecha = timezone.now() - timedelta(days=dias_atras)
            
            # Velocidad si es infracción de velocidad
            velocidad = None
            velocidad_max = None
            if tipo.codigo == 'EXCESO_VEL':
                velocidad_max = random.choice([40, 50, 60, 80, 100])
                velocidad = velocidad_max + random.randint(10, 40)
            
            infraccion = Infraccion.objects.create(
                vehiculo=vehiculo,
                tipo_infraccion=tipo,
                camara=camara,
                ubicacion=f"Zona {random.randint(1, 10)} - Calle {random.randint(1, 20)}",
                velocidad_detectada=velocidad,
                velocidad_maxima=velocidad_max,
                confianza_deteccion=random.uniform(85, 99),
                modelo_ia_version='YOLOv8n-demo',
                estado=random.choice(['DETECTADA', 'VERIFICADA', 'NOTIFICADA'])
            )
            infraccion.fecha_hora = fecha
            infraccion.save()
            
            infracciones_creadas += 1
    
    print(f"  ✅ {infracciones_creadas} infracciones creadas")
    
    return vehiculos

def ejecutar_predicciones_ml(vehiculos):
    """Ejecuta predicciones ML para todos los vehículos"""
    print("\n🧠 Ejecutando predicciones de Machine Learning...\n")
    
    predictor = PredictorRiesgo()
    
    resultados = []
    for vehiculo in vehiculos:
        try:
            resultado = predictor.predecir_riesgo_vehiculo(vehiculo.placa)
            resultados.append(resultado)
            
            print(f"📊 {vehiculo.placa}:")
            print(f"   Infracciones: {resultado['features']['total_infracciones']}")
            print(f"   Nivel de riesgo: {resultado['nivel_riesgo']}")
            print(f"   Probabilidad reincidencia: {resultado['probabilidad_reincidencia']:.1f}%")
            print()
            
        except Exception as e:
            print(f"❌ Error con {vehiculo.placa}: {e}")
    
    return resultados

def mostrar_estadisticas():
    """Muestra estadísticas del sistema"""
    print("\n" + "="*60)
    print("📈 ESTADÍSTICAS DEL SISTEMA")
    print("="*60)
    
    total_vehiculos = Vehiculo.objects.count()
    total_infracciones = Infraccion.objects.count()
    total_camaras = Camara.objects.count()
    total_perfiles = PerfilConductor.objects.count()
    
    print(f"\n🚗 Vehículos registrados: {total_vehiculos}")
    print(f"⚠️  Infracciones detectadas: {total_infracciones}")
    print(f"📹 Cámaras activas: {total_camaras}")
    print(f"👤 Perfiles de conductores: {total_perfiles}")
    
    # Infracciones por tipo
    print("\n📊 Infracciones por tipo:")
    tipos = TipoInfraccion.objects.all()
    for tipo in tipos:
        count = Infraccion.objects.filter(tipo_infraccion=tipo).count()
        if count > 0:
            print(f"   {tipo.nombre}: {count}")
    
    # Conductores de alto riesgo
    print("\n🚨 Conductores de alto riesgo:")
    perfiles_alto_riesgo = PerfilConductor.objects.filter(
        nivel_riesgo__in=['ALTO', 'CRITICO']
    ).order_by('-puntuacion_riesgo')[:5]
    
    for perfil in perfiles_alto_riesgo:
        print(f"   {perfil.vehiculo.placa}: {perfil.nivel_riesgo} "
              f"({perfil.probabilidad_reincidencia:.1f}% reincidencia)")
    
    print("\n" + "="*60)

def main():
    print("="*60)
    print("🎬 DEMOSTRACIÓN COMPLETA DEL SISTEMA")
    print("📚 Sistema de Detección de Infracciones con IA")
    print("="*60)
    print()
    
    # Crear datos de demostración
    vehiculos = crear_datos_demo()
    
    if not vehiculos:
        print("\n❌ No se pudieron crear datos de demostración")
        return
    
    # Ejecutar predicciones ML
    resultados = ejecutar_predicciones_ml(vehiculos)
    
    # Mostrar estadísticas
    mostrar_estadisticas()
    
    print("\n✅ Demostración completada exitosamente")
    print("\n📝 Próximos pasos:")
    print("   1. Abre el admin: python manage.py runserver")
    print("   2. Accede a: http://localhost:8000/admin")
    print("   3. Revisa las infracciones y perfiles creados")
    print("   4. Ejecuta el detector: python vision_ai/detector_webcam.py")
    print()

if __name__ == "__main__":
    main()

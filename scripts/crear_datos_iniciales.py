"""
Script para crear datos iniciales de prueba en el sistema
Ejecutar con: python manage.py shell < scripts/crear_datos_iniciales.py
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

# Ahora importar los modelos
from camaras.models import Camara
from infracciones.models import TipoInfraccion, Vehiculo, Infraccion, PerfilConductor
from django.utils import timezone
from datetime import timedelta
import random

print("🔄 Creando datos iniciales de prueba...")

# Crear cámaras
camaras_data = [
    {"ubicacion": "Av. Principal con Calle 5", "ip": "192.168.1.101", "descripcion": "Intersección principal"},
    {"ubicacion": "Av. Los Héroes 234", "ip": "192.168.1.102", "descripcion": "Zona comercial"},
    {"ubicacion": "Jr. Comercio 456", "ip": "192.168.1.103", "descripcion": "Centro de la ciudad"},
    {"ubicacion": "Av. Industrial 789", "ip": "192.168.1.104", "descripcion": "Zona industrial"},
]

camaras = []
for data in camaras_data:
    camara, created = Camara.objects.get_or_create(
        ubicacion=data["ubicacion"],
        defaults=data
    )
    camaras.append(camara)
    if created:
        print(f"✅ Cámara creada: {camara.ubicacion}")

# Verificar que existen tipos de infracción
tipos_infraccion = TipoInfraccion.objects.all()
if not tipos_infraccion:
    print("❌ No hay tipos de infracción. Ejecuta primero: python scripts/inicializar_sistema.py")
    sys.exit(1)

print(f"✅ Tipos de infracción disponibles: {tipos_infraccion.count()}")

# Crear vehículos de prueba
vehiculos_data = [
    {"placa": "ABC-123", "marca": "Toyota", "modelo": "Corolla", "color": "Blanco", "anio": 2020, "tipo_vehiculo": "AUTO"},
    {"placa": "XYZ-789", "marca": "Honda", "modelo": "Civic", "color": "Negro", "anio": 2019, "tipo_vehiculo": "AUTO"},
    {"placa": "DEF-456", "marca": "Yamaha", "modelo": "FZ", "color": "Rojo", "anio": 2021, "tipo_vehiculo": "MOTO"},
    {"placa": "GHI-321", "marca": "Nissan", "modelo": "Sentra", "color": "Azul", "anio": 2018, "tipo_vehiculo": "AUTO"},
    {"placa": "JKL-654", "marca": "Suzuki", "modelo": "Swift", "color": "Gris", "anio": 2022, "tipo_vehiculo": "AUTO"},
]

vehiculos = []
for data in vehiculos_data:
    vehiculo, created = Vehiculo.objects.get_or_create(
        placa=data["placa"], 
        defaults=data
    )
    vehiculos.append(vehiculo)
    if created:
        print(f"✅ Vehículo creado: {vehiculo.placa}")

# Crear infracciones de prueba
print("\n🔄 Creando infracciones de prueba...")
infracciones_creadas = 0

for i in range(20):
    vehiculo = random.choice(vehiculos)
    tipo = random.choice(list(tipos_infraccion))
    camara = random.choice(camaras)
    
    # Fecha aleatoria en los últimos 30 días
    dias_atras = random.randint(0, 30)
    fecha = timezone.now() - timedelta(days=dias_atras)
    
    infraccion_data = {
        "vehiculo": vehiculo,
        "tipo_infraccion": tipo,
        "camara": camara,
        "ubicacion": camara.ubicacion,
        "confianza_deteccion": round(random.uniform(85, 99), 2),
    }
    
    # Agregar datos específicos según tipo
    if tipo.codigo == "EXCESO_VEL":
        infraccion_data["velocidad_maxima"] = random.choice([40, 50, 60, 80])
        infraccion_data["velocidad_detectada"] = infraccion_data["velocidad_maxima"] + random.randint(10, 30)
    elif tipo.codigo == "LUZ_ROJA":
        infraccion_data["tiempo_luz_roja"] = round(random.uniform(0.5, 4.0), 2)
    
    infraccion = Infraccion.objects.create(**infraccion_data)
    infraccion.fecha_hora = fecha
    infraccion.save()
    infracciones_creadas += 1

print(f"✅ {infracciones_creadas} infracciones creadas")

# Crear perfiles de conductor
print("\n🔄 Creando perfiles de conductores...")
for vehiculo in vehiculos:
    perfil, created = PerfilConductor.objects.get_or_create(vehiculo=vehiculo)
    
    # Calcular estadísticas
    infracciones_vehiculo = vehiculo.infracciones.all()
    perfil.total_infracciones = infracciones_vehiculo.count()
    perfil.infracciones_luz_roja = infracciones_vehiculo.filter(tipo_infraccion__codigo="LUZ_ROJA").count()
    perfil.infracciones_velocidad = infracciones_vehiculo.filter(tipo_infraccion__codigo="EXCESO_VEL").count()
    perfil.infracciones_graves = infracciones_vehiculo.filter(tipo_infraccion__gravedad__in=["GRAVE", "MUY_GRAVE"]).count()
    
    # Calcular puntuación de riesgo simple
    puntuacion = (perfil.infracciones_graves * 20) + (perfil.total_infracciones * 5)
    perfil.puntuacion_riesgo = min(puntuacion, 100)
    
    if perfil.puntuacion_riesgo >= 75:
        perfil.nivel_riesgo = "CRITICO"
    elif perfil.puntuacion_riesgo >= 50:
        perfil.nivel_riesgo = "ALTO"
    elif perfil.puntuacion_riesgo >= 25:
        perfil.nivel_riesgo = "MEDIO"
    else:
        perfil.nivel_riesgo = "BAJO"
    
    perfil.save()
    print(f"✅ Perfil creado: {vehiculo.placa} - Riesgo {perfil.nivel_riesgo}")

print("\n" + "="*70)
print("✅ DATOS INICIALES CREADOS EXITOSAMENTE")
print("="*70)
print(f"📊 Resumen:")
print(f"   - Cámaras: {Camara.objects.count()}")
print(f"   - Tipos de infracciones: {TipoInfraccion.objects.count()}")
print(f"   - Vehículos: {Vehiculo.objects.count()}")
print(f"   - Infracciones: {Infraccion.objects.count()}")
print(f"   - Perfiles de conductores: {PerfilConductor.objects.count()}")
print("="*70)
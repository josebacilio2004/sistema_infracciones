"""
Script de inicialización del sistema completo
Crea datos iniciales y prepara el sistema para pruebas
"""
import os
import sys
import django
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

from infracciones.models import TipoInfraccion
from camaras.models import Camara

def crear_tipos_infracciones():
    """Crea el catálogo de tipos de infracciones"""
    print("📋 Creando catálogo de infracciones...")
    
    tipos = [
        {
            'codigo': 'LUZ_ROJA',
            'nombre': 'Pasarse luz roja',
            'descripcion': 'Cruzar intersección con semáforo en rojo',
            'monto_multa': 500.00,
            'puntos_licencia': 5,
            'gravedad': 'GRAVE'
        },
        {
            'codigo': 'EXCESO_VEL',
            'nombre': 'Exceso de velocidad',
            'descripcion': 'Superar el límite de velocidad permitido',
            'monto_multa': 300.00,
            'puntos_licencia': 3,
            'gravedad': 'MODERADA'
        },
        {
            'codigo': 'USO_CELULAR',
            'nombre': 'Uso de celular al conducir',
            'descripcion': 'Utilizar dispositivo móvil mientras conduce',
            'monto_multa': 200.00,
            'puntos_licencia': 2,
            'gravedad': 'MODERADA'
        },
        {
            'codigo': 'NO_CINTURON',
            'nombre': 'No usar cinturón de seguridad',
            'descripcion': 'Conducir sin cinturón de seguridad',
            'monto_multa': 150.00,
            'puntos_licencia': 2,
            'gravedad': 'LEVE'
        },
        {
            'codigo': 'ESTAC_PROHIB',
            'nombre': 'Estacionamiento prohibido',
            'descripcion': 'Estacionar en zona no permitida',
            'monto_multa': 100.00,
            'puntos_licencia': 1,
            'gravedad': 'LEVE'
        },
        {
            'codigo': 'GIRO_PROHIB',
            'nombre': 'Giro prohibido',
            'descripcion': 'Realizar giro en zona no permitida',
            'monto_multa': 250.00,
            'puntos_licencia': 3,
            'gravedad': 'MODERADA'
        },
        {
            'codigo': 'INVASION_CARRIL',
            'nombre': 'Invasión de carril',
            'descripcion': 'Invadir carril contrario o de emergencia',
            'monto_multa': 400.00,
            'puntos_licencia': 4,
            'gravedad': 'GRAVE'
        },
        {
            'codigo': 'NO_PASO_PEAT',
            'nombre': 'No respetar paso peatonal',
            'descripcion': 'No ceder el paso a peatones',
            'monto_multa': 350.00,
            'puntos_licencia': 4,
            'gravedad': 'GRAVE'
        }
    ]
    
    creados = 0
    for tipo_data in tipos:
        tipo, created = TipoInfraccion.objects.get_or_create(
            codigo=tipo_data['codigo'],
            defaults=tipo_data
        )
        if created:
            creados += 1
            print(f"  ✅ {tipo.nombre}")
    
    print(f"\n✅ {creados} tipos de infracciones creados")

def crear_camaras_prueba():
    """Crea cámaras de prueba"""
    print("\n📹 Creando cámaras de prueba...")
    
    camaras = [
        {
            'ubicacion': 'Webcam Local - Pruebas',
            'ip': '127.0.0.1',
            'descripcion': 'Cámara de prueba con webcam'
        },
        {
            'ubicacion': 'Av. Principal - Intersección Norte',
            'ip': '192.168.1.100',
            'descripcion': 'Cámara en intersección principal'
        },
        {
            'ubicacion': 'Calle Central - Zona Escolar',
            'ip': '192.168.1.101',
            'descripcion': 'Cámara en zona escolar'
        }
    ]
    
    creados = 0
    for cam_data in camaras:
        cam, created = Camara.objects.get_or_create(
            ubicacion=cam_data['ubicacion'],
            defaults=cam_data
        )
        if created:
            creados += 1
            print(f"  ✅ {cam.ubicacion}")
    
    print(f"\n✅ {creados} cámaras creadas")

def main():
    print("=" * 60)
    print("🚀 INICIALIZACIÓN DEL SISTEMA")
    print("📚 Proyecto de Tesis - Sistema de Infracciones con IA")
    print("=" * 60)
    print()
    
    crear_tipos_infracciones()
    crear_camaras_prueba()
    
    print("\n" + "=" * 60)
    print("✅ SISTEMA INICIALIZADO CORRECTAMENTE")
    print("=" * 60)
    print("\n📝 Próximos pasos:")
    print("   1. python manage.py migrate")
    print("   2. python manage.py createsuperuser")
    print("   3. python scripts/entrenar_modelo_ml.py")
    print("   4. python vision_ai/detector_webcam.py")
    print()

if __name__ == "__main__":
    main()

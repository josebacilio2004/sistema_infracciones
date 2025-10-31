"""
Diagnóstico del detector en entorno Django
"""
import os
import sys
import django
import cv2
import numpy as np

# Configurar Django
sys.path.append('/ruta/a/tu/proyecto')  # Ajusta la ruta
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

def diagnostico_completo():
    print("🔍 DIAGNÓSTICO COMPLETO DEL SERVIDOR")
    print("=" * 60)
    
    # 1. Verificar imports
    try:
        from vision_ai.detector_pytesseract import DetectorPytesseract
        print("✅ DetectorPytesseract importado correctamente")
    except Exception as e:
        print(f"❌ Error importando detector: {e}")
        return
    
    # 2. Crear imagen de prueba
    img_prueba = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.putText(img_prueba, "ABC-123", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # 3. Probar detector
    try:
        print("🚀 Inicializando detector...")
        detector = DetectorPytesseract()
        
        print("🔍 Procesando imagen de prueba...")
        resultado, detecciones = detector.procesar_frame(img_prueba)
        
        print(f"✅ Procesamiento exitoso")
        print(f"   - Frame resultado: {type(resultado)}")
        print(f"   - Detecciones: {len(detecciones)}")
        
        # Verificar placas detectadas
        if hasattr(detector, 'placas_detectadas'):
            print(f"   - Placas detectadas: {len(detector.placas_detectadas)}")
            for placa in detector.placas_detectadas:
                print(f"     🚗 {placa.get('placa', 'N/A')}")
        else:
            print("   - No se encontró atributo 'placas_detectadas'")
            
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnostico_completo()
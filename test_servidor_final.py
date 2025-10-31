"""
Prueba FINAL del detector en entorno servidor - VERSIÓN CORREGIDA
"""
import os
import sys
import django
import cv2
import numpy as np

# Configurar Django - USA LA RUTA ABSOLUTA DE TU PROYECTO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

# 🔥 CONFIGURAR PYTESSERACT ANTES DE CUALQUIER IMPORT
try:
    import pytesseract
    # ✅ CONFIGURACIÓN PARA TU SISTEMA
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Bacilio\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    PYTESSERACT_DISPONIBLE = True
    print("✅ pytesseract configurado correctamente en test")
except Exception as e:
    print(f"❌ Error configurando pytesseract: {e}")
    PYTESSERACT_DISPONIBLE = False

def prueba_ocr_directo():
    """Prueba solo el OCR sin YOLO - VERSIÓN CORREGIDA"""
    print("\n🔍 PRUEBA OCR DIRECTO")
    print("=" * 50)
    
    if not PYTESSERACT_DISPONIBLE:
        print("❌ pytesseract no disponible - saltando prueba OCR")
        return
    
    try:
        # Crear imagen con texto claro
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, "XYZ-789", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Probar OCR directo
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Verificar que Tesseract funcione
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version: {version}")
        except:
            print("❌ No se pudo obtener versión de Tesseract")
            return
        
        texto = pytesseract.image_to_string(gray, config='--psm 8')
        print(f"📝 OCR result: '{texto.strip()}'")
        
        # Probar con diferentes configuraciones
        configs = [
            '--psm 8',
            '--psm 7', 
            '--psm 13',
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        ]
        
        for config in configs:
            try:
                texto = pytesseract.image_to_string(gray, config=config)
                print(f"   Config {config}: '{texto.strip()}'")
            except Exception as e:
                print(f"   Config {config}: ERROR - {e}")
                
    except Exception as e:
        print(f"❌ Error en prueba OCR directo: {e}")

def prueba_con_imagen_real():
    """Prueba con una imagen que tiene texto de placa"""
    print("\n🧪 PRUEBA CON IMAGEN REAL")
    print("=" * 50)
    
    try:
        from vision_ai.detector_pytesseract import DetectorPytesseract
    except Exception as e:
        print(f"❌ Error importando detector: {e}")
        return
    
    # Crear imagen de prueba REALISTA
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Dibujar rectángulo de placa
    cv2.rectangle(img, (150, 280), (450, 340), (255, 255, 255), -1)
    cv2.rectangle(img, (150, 280), (450, 340), (0, 0, 0), 2)
    
    # Texto de placa CLARO Y LEGIBLE
    cv2.putText(img, "ABC-123", (180, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    
    # Inicializar detector
    print("🚀 Inicializando detector...")
    try:
        detector = DetectorPytesseract()
    except Exception as e:
        print(f"❌ Error inicializando detector: {e}")
        return
    
    # Procesar múltiples veces
    for i in range(3):
        print(f"\n🔄 Iteración {i+1}:")
        try:
            resultado, detecciones = detector.procesar_frame(img)
            
            print(f"   - Vehículos YOLO: {len(detecciones)}")
            print(f"   - Placas detectadas: {len(detector.placas_detectadas)}")
            
            for placa in detector.placas_detectadas:
                print(f"   🎯 PLACA: {placa.get('placa')}")
                
        except Exception as e:
            print(f"❌ Error en iteración {i+1}: {e}")
    
    # Guardar imagen de prueba
    cv2.imwrite('test_servidor.jpg', img)
    print(f"\n💾 Imagen guardada: 'test_servidor.jpg'")

def prueba_sin_dependencias():
    """Prueba básica sin dependencias externas"""
    print("\n🧪 PRUEBA BÁSICA SIN DEPENDENCIAS")
    print("=" * 50)
    
    # Crear imagen simple
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST-456", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print(f"📐 Imagen creada: {img.shape}")
    print("✅ Prueba básica completada")
    
    # Guardar para verificar
    cv2.imwrite('test_basico.jpg', img)
    print("💾 Imagen básica guardada: 'test_basico.jpg'")

if __name__ == "__main__":
    print("🚀 PRUEBA DEFINITIVA ENTORNO SERVIDOR")
    print("=" * 60)
    
    # 1. Prueba básica primero
    prueba_sin_dependencias()
    
    # 2. Prueba OCR directo (si está disponible)
    if PYTESSERACT_DISPONIBLE:
        prueba_ocr_directo()
    else:
        print("\n⚠️  pytesseract no disponible - saltando pruebas OCR")
    
    # 3. Prueba con detector completo
    prueba_con_imagen_real()
    
    print("\n✅ PRUEBAS COMPLETADAS")
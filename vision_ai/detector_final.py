"""
DETECTOR FINAL con pytesseract - Configurado para tu sistema funcionando
"""
import cv2
import numpy as np
import threading
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import deque
import sys
from pathlib import Path

# Configurar rutas
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Importar utilidades de placas
try:
    from vision_ai.utils_placas import (
        normalizar_placa_peruana, 
        validar_placa_peruana,
        limpiar_texto_placa
    )
    print("✅ Utils_placas cargado correctamente")
except ImportError as e:
    print(f"⚠️ No se pudo cargar utils_placas: {e}")

# ✅ CONFIGURACIÓN CORRECTA PARA TU TESSERACT FUNCIONANDO
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Bacilio\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    print("✅ Tesseract v5.5.0 configurado correctamente")
    
    # Verificar que funciona
    test_version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract version: {test_version}")
    
except Exception as e:
    print(f"❌ Error configurando Tesseract: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

class DetectorFinal:
    """Detector FINAL con pytesseract funcionando"""
    
    def __init__(self, camara_id=None, **kwargs):
        print("=" * 60)
        print("🚀 INICIANDO DETECTOR FINAL CON TESSERACT v5.5.0")
        print("=" * 60)
        
        self.camara_id = camara_id
        self.frame_count = 0
        self.detecciones_vehiculos = []
        self.placas_detectadas = []
        self.lock = threading.Lock()
        self.running = True
        
        # Configuración optimizada
        self.confianza_minima_vehiculos = 0.3  # Más bajo para detectar más vehículos
        self.ocr_cada_frames = 3  # OCR frecuente
        
        # Métricas
        self.fps_real = deque(maxlen=30)
        self.tiempo_inicio = time.time()
        
        # Cargar modelos
        self._cargar_modelos_final()
        
        print("✅ DETECTOR FINAL INICIALIZADO CORRECTAMENTE")
        print(f"   - Tesseract v5.5.0")
        print(f"   - YOLOv8n optimizado") 
        print(f"   - OCR cada {self.ocr_cada_frames} frames")
        print("=" * 60)
    
    def _cargar_modelos_final(self):
        """Carga todos los modelos necesarios"""
        try:
            # ✅ YOLOv8
            from ultralytics import YOLO
            print("📦 Cargando YOLOv8n...")
            self.modelo_yolo = YOLO('yolov8n.pt')
            self.modelo_yolo.fuse()
            print("✅ YOLOv8 cargado y optimizado")
            
        except Exception as e:
            print(f"❌ Error crítico cargando YOLO: {e}")
            raise
    
    def procesar_frame(self, frame) -> Tuple[np.ndarray, List[Dict]]:
        """
        Procesa frame completo - Compatible con API Django
        """
        with self.lock:
            if not self.running:
                return frame, []
            
            self.frame_count += 1
            
            # Calcular FPS
            tiempo_actual = time.time()
            fps_actual = 1.0 / (tiempo_actual - self.tiempo_inicio + 1e-6)
            self.fps_real.append(fps_actual)
            self.tiempo_inicio = tiempo_actual
            
            # 1. DETECTAR VEHÍCULOS (siempre)
            vehiculos = self._detectar_vehiculos_mejorado(frame)
            self.detecciones_vehiculos = vehiculos
            
            # 2. BUSCAR PLACAS (condicional - optimizado)
            placas = []
            if vehiculos and (self.frame_count % self.ocr_cada_frames == 0):
                placas = self._buscar_placas_todos_vehiculos(frame, vehiculos)
                self.placas_detectadas = placas
            
            # 3. DIBUJAR RESULTADOS
            frame_anotado = self._dibujar_resultados_completos(frame, vehiculos, placas)
            
            # 4. LOG DE RESULTADOS
            if placas:
                for placa in placas:
                    print(f"🎯 PLACA DETECTADA: {placa['placa']} (conf: {placa['confidence']:.2f})")
            
            return frame_anotado, vehiculos
    
    def _detectar_vehiculos_mejorado(self, frame) -> List[Dict]:
        """Detección mejorada de vehículos"""
        try:
            detecciones = []
            
            # Configuración optimizada para máxima detección
            resultados = self.modelo_yolo(
                frame, 
                verbose=False,
                conf=self.confianza_minima_vehiculos,  # Bajo para detectar más
                iou=0.4,  # Menor NMS para más detecciones
                classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
                agnostic_nms=True  # Mejor para múltiples clases
            )
            
            for resultado in resultados:
                for box in resultado.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confianza = float(box.conf[0].cpu().numpy())
                    clase_id = int(box.cls[0].cpu().numpy())
                    
                    # Solo agregar si es suficientemente grande (evitar falsos positivos pequeños)
                    ancho = x2 - x1
                    alto = y2 - y1
                    area = ancho * alto
                    
                    if area > 1000:  # Mínimo 1000 píxeles de área
                        detecciones.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confianza,
                            'class_id': clase_id,
                            'class_name': self._get_class_name(clase_id),
                            'timestamp': datetime.now(),
                            'area': area
                        })
            
            return detecciones
            
        except Exception as e:
            logger.error(f"Error en detección de vehículos: {e}")
            return []
    
    def _buscar_placas_todos_vehiculos(self, frame, vehiculos) -> List[Dict]:
        """Busca placas en todos los vehículos detectados"""
        placas_encontradas = []
        
        for vehiculo in vehiculos:
            # Priorizar vehículos más grandes y con mayor confianza
            if vehiculo['confidence'] > 0.5 and vehiculo['area'] > 5000:
                placa = self._procesar_placa_vehiculo(frame, vehiculo)
                if placa:
                    placas_encontradas.append(placa)
                    # No buscar más si ya encontramos una placa (para rendimiento)
                    break
        
        return placas_encontradas
    
    def _procesar_placa_vehiculo(self, frame, vehiculo) -> Optional[Dict]:
        """Procesa un vehículo específico para buscar placa"""
        try:
            x1, y1, x2, y2 = vehiculo['bbox']
            roi_vehiculo = frame[y1:y2, x1:x2]
            
            if roi_vehiculo.size == 0:
                return None
            
            # ESTRATEGIAS MÚLTIPLES para encontrar la placa
            estrategias = [
                self._buscar_placa_parte_inferior,
                self._buscar_placa_centro_inferior, 
                self._buscar_placa_región_completa
            ]
            
            for estrategia in estrategias:
                placa = estrategia(roi_vehiculo, x1, y1)
                if placa:
                    return placa
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error procesando vehículo: {e}")
            return None
    
    def _buscar_placa_parte_inferior(self, roi_vehiculo, offset_x, offset_y) -> Optional[Dict]:
        """Busca placa en la parte inferior del vehículo"""
        try:
            h, w = roi_vehiculo.shape[:2]
            
            # Región inferior (donde suelen estar las placas)
            y_start = int(h * 0.7)  # 70% desde arriba
            y_end = int(h * 0.95)   # 95% desde arriba  
            x_start = int(w * 0.1)  # 10% desde izquierda
            x_end = int(w * 0.9)    # 90% desde izquierda
            
            region_placa = roi_vehiculo[y_start:y_end, x_start:x_end]
            
            if region_placa.size == 0:
                return None
            
            return self._ejecutar_ocr_y_validar(region_placa, offset_x + x_start, offset_y + y_start)
            
        except Exception as e:
            return None
    
    def _buscar_placa_centro_inferior(self, roi_vehiculo, offset_x, offset_y) -> Optional[Dict]:
        """Busca placa en el centro-inferior"""
        try:
            h, w = roi_vehiculo.shape[:2]
            
            # Región centro-inferior
            y_start = int(h * 0.6)
            y_end = int(h * 0.9)
            x_start = int(w * 0.25) 
            x_end = int(w * 0.75)
            
            region_placa = roi_vehiculo[y_start:y_end, x_start:x_end]
            
            if region_placa.size == 0:
                return None
            
            return self._ejecutar_ocr_y_validar(region_placa, offset_x + x_start, offset_y + y_start)
            
        except Exception as e:
            return None
    
    def _buscar_placa_región_completa(self, roi_vehiculo, offset_x, offset_y) -> Optional[Dict]:
        """Busca placa en toda la región del vehículo (último recurso)"""
        try:
            return self._ejecutar_ocr_y_validar(roi_vehiculo, offset_x, offset_y)
        except Exception as e:
            return None
    
    def _ejecutar_ocr_y_validar(self, imagen_region, offset_x, offset_y) -> Optional[Dict]:
        """Ejecuta OCR y valida si es una placa peruana válida"""
        try:
            # Preprocesamiento AGGRESIVO para OCR
            imagen_procesada = self._preprocesar_agresivo_ocr(imagen_region)
            
            # Múltiples configuraciones PSM
            configuraciones_psm = [
                '--psm 8',  # Palabra única
                '--psm 7',  # Línea única
                '--psm 13', # Línea con segmentación
                '--psm 6',  # Bloque uniforme
            ]
            
            for psm in configuraciones_psm:
                try:
                    config = f'{psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    texto = pytesseract.image_to_string(imagen_procesada, config=config)
                    
                    texto_limpio = limpiar_texto_placa(texto)
                    if not texto_limpio:
                        continue
                    
                    placa_normalizada = normalizar_placa_peruana(texto_limpio)
                    if placa_normalizada and validar_placa_peruana(placa_normalizada):
                        print(f"🔍 OCR EXITOSO: '{texto_limpio}' -> '{placa_normalizada}' (PSM: {psm})")
                        
                        h, w = imagen_region.shape[:2]
                        return {
                            'placa': placa_normalizada,
                            'bbox': (offset_x, offset_y, w, h),
                            'confidence': 0.85,  # Confianza alta para placas válidas
                            'timestamp': datetime.now(),
                            'estrategia': 'pytesseract'
                        }
                        
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            print(f"❌ Error en OCR: {e}")
            return None
    
    def _preprocesar_agresivo_ocr(self, imagen):
        """Preprocesamiento AGGRESIVO para mejorar OCR"""
        try:
            # Convertir a escala de grises
            if len(imagen.shape) == 3:
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                gray = imagen
            
            # 1. Redimensionar si es muy pequeña
            h, w = gray.shape
            if h < 50 or w < 100:
                new_h, new_w = max(50, h), max(100, w)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # 2. Mejorar contraste con CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # 3. Suavizado y umbral
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 4. Operaciones morfológicas para limpiar
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            print(f"❌ Error en preprocesamiento: {e}")
            return imagen
    
    def _get_class_name(self, class_id: int) -> str:
        """Nombre de clases en español"""
        class_names = {
            2: 'carro',
            3: 'moto', 
            5: 'bus',
            7: 'camion'
        }
        return class_names.get(class_id, f'clase_{class_id}')
    
    def _dibujar_resultados_completos(self, frame, vehiculos, placas) -> np.ndarray:
        """Dibuja todos los resultados en el frame"""
        frame_anotado = frame.copy()
        
        # DIBUJAR VEHÍCULOS (verde)
        for vehiculo in vehiculos:
            x1, y1, x2, y2 = vehiculo['bbox']
            color = (0, 255, 0)  # Verde
            
            cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color, 2)
            
            etiqueta = f"{vehiculo['class_name']} {vehiculo['confidence']:.2f}"
            cv2.putText(frame_anotado, etiqueta, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # DIBUJAR PLACAS (rojo - más prominente)
        for placa in placas:
            x, y, w, h = placa['bbox']
            color = (0, 0, 255)  # Rojo
            
            # Rectángulo grueso para placa
            cv2.rectangle(frame_anotado, (x, y), (x+w, y+h), color, 3)
            
            # Texto destacado
            etiqueta_placa = f"PLACA: {placa['placa']}"
            cv2.putText(frame_anotado, etiqueta_placa, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        
        # PANEL DE INFORMACIÓN
        fps_promedio = sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0
        
        # Fondo semitransparente
        overlay = frame_anotado.copy()
        cv2.rectangle(overlay, (5, 5), (350, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_anotado, 0.3, 0, frame_anotado)
        
        # Borde
        cv2.rectangle(frame_anotado, (5, 5), (350, 110), (0, 255, 0), 2)
        
        # Textos informativos
        cv2.putText(frame_anotado, f"FPS: {fps_promedio:.1f}", (15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_anotado, f"Vehículos: {len(vehiculos)}", (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_anotado, f"Placas: {len(placas)}", (15, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_anotado, f"Frame: {self.frame_count}", (15, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Estado del OCR
        estado_ocr = "🟢 ACTIVO" if len(placas) > 0 else "🟡 BUSCANDO"
        cv2.putText(frame_anotado, f"OCR: {estado_ocr}", (180, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame_anotado
    
    def get_estadisticas(self) -> Dict:
        """Estadísticas para la API"""
        return {
            'total_frames': self.frame_count,
            'vehiculos_detectados': len(self.detecciones_vehiculos),
            'placas_detectadas': len(self.placas_detectadas),
            'fps_promedio': sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0,
            'detector': 'Final con Tesseract v5.5.0'
        }
    
    def detener(self):
        """Detener el detector"""
        self.running = False
        print("🛑 DetectorFinal detenido")


# Prueba rápida
if __name__ == "__main__":
    print("🧪 PRUEBA RÁPIDA DEL DETECTOR FINAL")
    
    # Probar con imagen si existe
    import os
    if os.path.exists('placa_test.jpg'):
        detector = DetectorFinal()
        frame = cv2.imread('placa_test.jpg')
        if frame is not None:
            resultado, detecciones = detector.procesar_frame(frame)
            print(f"✅ Procesado - Vehículos: {len(detecciones)}, Placas: {len(detector.placas_detectadas)}")
            cv2.imwrite('prueba_final.jpg', resultado)
    else:
        print("⚠️  Crea 'placa_test.jpg' para probar")
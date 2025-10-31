"""
Detector optimizado para mejor rendimiento con EasyOCR
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

logger = logging.getLogger(__name__)

class DetectorOptimizado:
    """Detector optimizado para mejor rendimiento"""
    
    def __init__(self, camara_id=None, **kwargs):
        print("🚀 Inicializando DetectorOptimizado...")
        
        self.camara_id = camara_id
        self.frame_count = 0
        self.detecciones_vehiculos = []
        self.placas_detectadas = []
        self.lock = threading.Lock()
        self.running = False
        
        # Métricas
        self.fps_real = deque(maxlen=30)
        self.tiempo_inicio = time.time()
        
        # Cargar modelos de forma optimizada
        self._cargar_modelos_optimizados()
        
        print("✅ DetectorOptimizado inicializado correctamente")
    
    def _cargar_modelos_optimizados(self):
        """Carga modelos de forma optimizada"""
        try:
            # ✅ YOLOv8 (rápido)
            from ultralytics import YOLO
            print("📦 Cargando YOLOv8n...")
            self.modelo_yolo = YOLO('yolov8n.pt')
            self.modelo_yolo.fuse()
            print("✅ YOLOv8 cargado correctamente")
            
            # ✅ EasyOCR con configuración optimizada
            try:
                import easyocr
                print("📝 Cargando EasyOCR (puede tomar unos segundos)...")
                
                # Configuración optimizada para CPU
                self.reader = easyocr.Reader(
                    ['en'], 
                    gpu=False,  # Forzar CPU para mayor estabilidad
                    download_enabled=True,
                    model_storage_directory='./easyocr_models',
                    verbose=False  # Menos logs
                )
                print("✅ EasyOCR inicializado (CPU)")
                
            except ImportError as e:
                print(f"⚠️ EasyOCR no disponible: {e}")
                self.reader = None
            except Exception as e:
                print(f"❌ Error inicializando EasyOCR: {e}")
                self.reader = None
            
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            self.modelo_yolo = None
            self.reader = None
    
    def procesar_frame(self, frame) -> Tuple[np.ndarray, List[Dict]]:
        """Procesa frame de forma optimizada"""
        with self.lock:
            self.frame_count += 1
            
            # Calcular FPS
            tiempo_actual = time.time()
            self.fps_real.append(1 / (tiempo_actual - self.tiempo_inicio + 1e-6))
            self.tiempo_inicio = tiempo_actual
            
            # 1. Detectar vehículos (siempre rápido)
            vehiculos = self._detectar_vehiculos_rapido(frame)
            self.detecciones_vehiculos = vehiculos
            
            # 2. Solo buscar placas si hay vehículos y cada 5 frames (para rendimiento)
            placas = []
            if vehiculos and self.frame_count % 5 == 0 and self.reader is not None:
                placas = self._buscar_placas_optimizado(frame, vehiculos)
            
            self.placas_detectadas = placas
            
            # 3. Dibujar anotaciones
            frame_anotado = self._dibujar_anotaciones_optimizadas(frame, vehiculos, placas)
            
            return frame_anotado, vehiculos
    
    def _detectar_vehiculos_rapido(self, frame) -> List[Dict]:
        """Detección rápida de vehículos"""
        if self.modelo_yolo is None:
            return []
        
        try:
            detecciones = []
            
            # Configuración optimizada para velocidad
            resultados = self.modelo_yolo(
                frame, 
                verbose=False,
                conf=0.5,  # Umbral de confianza
                iou=0.5,   # Umbral de NMS
                classes=[2, 3, 5, 7]  # Solo vehículos
            )
            
            for resultado in resultados:
                for box in resultado.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confianza = box.conf[0].cpu().numpy()
                    clase_id = int(box.cls[0].cpu().numpy())
                    
                    detecciones.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confianza),
                        'class_id': clase_id,
                        'class_name': self._get_class_name(clase_id),
                        'timestamp': datetime.now()
                    })
            
            return detecciones
        
        except Exception as e:
            logger.error(f"Error detectando vehículos: {e}")
            return []
    
    def _buscar_placas_optimizado(self, frame, vehiculos) -> List[Dict]:
        """Búsqueda optimizada de placas"""
        placas_encontradas = []
        
        for vehiculo in vehiculos[:2]:  # Solo procesar primeros 2 vehículos
            placa = self._procesar_placa_vehiculo(frame, vehiculo)
            if placa:
                placas_encontradas.append(placa)
        
        return placas_encontradas
    
    def _procesar_placa_vehiculo(self, frame, vehiculo) -> Optional[Dict]:
        """Procesa placa de un vehículo específico"""
        try:
            x1, y1, x2, y2 = vehiculo['bbox']
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # Estrategia 1: Parte inferior del vehículo
            placa = self._buscar_placa_parte_inferior(roi, x1, y1)
            if placa:
                return placa
            
            # Estrategia 2: Región completa (más lenta)
            placa = self._buscar_placa_region_completa(roi, x1, y1)
            return placa
            
        except Exception as e:
            logger.error(f"Error procesando placa: {e}")
            return None
    
    def _buscar_placa_parte_inferior(self, roi, offset_x, offset_y) -> Optional[Dict]:
        """Busca placa en la parte inferior (más rápida)"""
        try:
            altura_roi = roi.shape[0]
            ancho_roi = roi.shape[1]
            
            # Solo buscar en parte inferior (60% - 90%)
            y_start = int(altura_roi * 0.6)
            y_end = int(altura_roi * 0.9)
            x_start = int(ancho_roi * 0.2)
            x_end = int(ancho_roi * 0.8)
            
            region_placa = roi[y_start:y_end, x_start:x_end]
            
            if region_placa.size == 0:
                return None
            
            # Preprocesar para mejor OCR
            region_procesada = self._preprocesar_imagen_ocr(region_placa)
            
            # OCR con timeout
            resultados = self.reader.readtext(
                region_procesada,
                detail=1,
                paragraph=False,
                text_threshold=0.4,  # Más bajo para capturar más
                low_text=0.3,
                link_threshold=0.4,
                width_ths=0.7,
                height_ths=0.7,
                batch_size=1  # Procesar una a la vez
            )
            
            for (bbox, texto, confianza) in resultados:
                texto_limpio = limpiar_texto_placa(texto)
                placa_normalizada = normalizar_placa_peruana(texto_limpio)
                
                if placa_normalizada and validar_placa_peruana(placa_normalizada) and confianza > 0.3:
                    # Convertir coordenadas
                    (x_p1, y_p1), (x_p2, y_p2), (x_p3, y_p3), (x_p4, y_p4) = bbox
                    
                    x_abs = offset_x + x_start + int(min(x_p1, x_p2, x_p3, x_p4))
                    y_abs = offset_y + y_start + int(min(y_p1, y_p2, y_p3, y_p4))
                    w_abs = int(max(x_p1, x_p2, x_p3, x_p4) - min(x_p1, x_p2, x_p3, x_p4))
                    h_abs = int(max(y_p1, y_p2, y_p3, y_p4) - min(y_p1, y_p2, y_p3, y_p4))
                    
                    print(f"🚗 Placa detectada: {placa_normalizada} (conf: {confianza:.2f})")
                    
                    return {
                        'placa': placa_normalizada,
                        'bbox': (x_abs, y_abs, w_abs, h_abs),
                        'confidence': float(confianza),
                        'timestamp': datetime.now(),
                        'estrategia': 'parte_inferior'
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Error en búsqueda parte inferior: {e}")
            return None
    
    def _buscar_placa_region_completa(self, roi, offset_x, offset_y) -> Optional[Dict]:
        """Búsqueda en región completa (más lenta, usar solo si es necesario)"""
        try:
            region_procesada = self._preprocesar_imagen_ocr(roi)
            
            resultados = self.reader.readtext(
                region_procesada,
                detail=1,
                paragraph=False,
                text_threshold=0.4,
                batch_size=1
            )
            
            for (bbox, texto, confianza) in resultados:
                texto_limpio = limpiar_texto_placa(texto)
                placa_normalizada = normalizar_placa_peruana(texto_limpio)
                
                if placa_normalizada and validar_placa_peruana(placa_normalizada) and confianza > 0.3:
                    (x_p1, y_p1), (x_p2, y_p2), (x_p3, y_p3), (x_p4, y_p4) = bbox
                    
                    x_abs = offset_x + int(min(x_p1, x_p2, x_p3, x_p4))
                    y_abs = offset_y + int(min(y_p1, y_p2, y_p3, y_p4))
                    w_abs = int(max(x_p1, x_p2, x_p3, x_p4) - min(x_p1, x_p2, x_p3, x_p4))
                    h_abs = int(max(y_p1, y_p2, y_p3, y_p4) - min(y_p1, y_p2, y_p3, y_p4))
                    
                    print(f"🚗 Placa detectada (región): {placa_normalizada} (conf: {confianza:.2f})")
                    
                    return {
                        'placa': placa_normalizada,
                        'bbox': (x_abs, y_abs, w_abs, h_abs),
                        'confidence': float(confianza),
                        'timestamp': datetime.now(),
                        'estrategia': 'region_completa'
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Error en búsqueda región completa: {e}")
            return None
    
    def _preprocesar_imagen_ocr(self, imagen):
        """Preprocesamiento optimizado para OCR"""
        try:
            # Convertir a escala de grises
            if len(imagen.shape) == 3:
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                gray = imagen
            
            # Mejorar contraste rápido
            gray = cv2.medianBlur(gray, 3)
            
            # CLAHE para mejor contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Binarización adaptativa
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return binary
            
        except Exception as e:
            print(f"❌ Error en preprocesamiento: {e}")
            return imagen
    
    def _get_class_name(self, class_id: int) -> str:
        """Obtiene nombre de la clase"""
        class_names = {
            2: 'carro',
            3: 'moto', 
            5: 'bus',
            7: 'camion'
        }
        return class_names.get(class_id, f'clase_{class_id}')
    
    def _dibujar_anotaciones_optimizadas(self, frame, vehiculos, placas) -> np.ndarray:
        """Dibuja anotaciones optimizadas"""
        frame_anotado = frame.copy()
        
        # Dibujar vehículos
        for vehiculo in vehiculos:
            x1, y1, x2, y2 = vehiculo['bbox']
            cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            etiqueta = f"{vehiculo['class_name']} ({vehiculo['confidence']:.2f})"
            cv2.putText(frame_anotado, etiqueta,
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Dibujar placas
        for placa in placas:
            x, y, w, h = placa['bbox']
            cv2.rectangle(frame_anotado, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            etiqueta_placa = f"{placa['placa']} ({placa['confidence']:.2f})"
            cv2.putText(frame_anotado, etiqueta_placa,
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Información de rendimiento
        fps_promedio = sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0
        
        cv2.putText(frame_anotado, f"FPS: {fps_promedio:.1f} | Vehículos: {len(vehiculos)} | Placas: {len(placas)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_anotado
    
    def get_estadisticas(self) -> Dict:
        """Obtiene estadísticas"""
        return {
            'total_frames': self.frame_count,
            'vehiculos_detectados': len(self.detecciones_vehiculos),
            'placas_detectadas': len(self.placas_detectadas),
            'fps_promedio': sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0
        }
    
    def detener(self):
        """Detiene el detector"""
        self.running = False
        print("🛑 DetectorOptimizado detenido")
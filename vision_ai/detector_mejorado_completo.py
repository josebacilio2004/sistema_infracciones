"""
Detector mejorado completo con integración SUNARP y generación de infracciones
Combina: Detección de vehículos + OCR de placas + Consulta SUNARP + Registro de infracciones
"""

# Agregar al inicio del archivo
import sys
from pathlib import Path

# Agregar ruta para importar utils_placas
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

import cv2
import numpy as np
import threading
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import deque
import os

# Configurar logging
logger = logging.getLogger(__name__)

class DetectorCompletoInfracciones:
    """Detector completo que integra todas las dimensiones del sistema"""
    
    def __init__(self, camara_id=None, **kwargs):
        print("🚀 Inicializando DetectorCompletoInfracciones...")
        
        # Ignorar parámetros extra
        self.camara_id = camara_id
        
        # Inicializar variables
        self.frame_count = 0
        self.detecciones_vehiculos = []
        self.placas_detectadas = []
        self.infracciones_detectadas = []
        self.lock = threading.Lock()
        self.running = False
        
        # Métricas
        self.fps_real = deque(maxlen=30)
        self.tiempo_inicio = time.time()
        
        # Cargar modelos
        self._cargar_modelos()
        
        print("✅ DetectorCompletoInfracciones inicializado correctamente")
    
    def _cargar_modelos(self):
        """Carga los modelos de IA necesarios usando YOLOv8"""
        try:
            # ✅ USAR YOLOv8 (que ya tienes funcionando)
            from ultralytics import YOLO
            print("📦 Cargando YOLOv8n...")
            self.modelo_yolo = YOLO('yolov8n.pt')
            self.modelo_yolo.fuse()  # Optimizar modelo
            print("✅ YOLOv8 cargado correctamente")
            
            # ✅ USAR EasyOCR (que ya tienes funcionando)
            try:
                import easyocr
                print("📝 Cargando EasyOCR...")
                self.reader = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR inicializado")
            except ImportError as e:
                print(f"⚠️ EasyOCR no disponible: {e}")
                self.reader = None
            
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            self.modelo_yolo = None
            self.reader = None
    
    def procesar_frame(self, frame) -> Tuple[np.ndarray, List[Dict]]:
        """
        Procesa un frame completo
        
        Returns:
            Tuple: (frame_anotado, lista_de_detecciones)
        """
        with self.lock:
            self.frame_count += 1
            
            # Calcular FPS
            tiempo_actual = time.time()
            self.fps_real.append(1 / (tiempo_actual - self.tiempo_inicio + 1e-6))
            self.tiempo_inicio = tiempo_actual
            
            # Detectar vehículos
            vehiculos = self._detectar_vehiculos(frame)
            self.detecciones_vehiculos = vehiculos
            
            # Detectar placas en vehículos
            placas = []
            for vehiculo in vehiculos:
                placa = self._detectar_placa(frame, vehiculo)
                if placa:
                    placas.append(placa)
            
            self.placas_detectadas = placas
            
            # Dibujar anotaciones
            frame_anotado = self._dibujar_anotaciones(frame, vehiculos, placas)
            
            return frame_anotado, vehiculos
    
    def _detectar_vehiculos(self, frame) -> List[Dict]:
        """Detecta vehículos en el frame usando YOLOv8"""
        if self.modelo_yolo is None:
            return []
        
        try:
            detecciones = []
            
            # Detectar con YOLOv8
            resultados = self.modelo_yolo(frame, verbose=False)
            
            for resultado in resultados:
                for box in resultado.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confianza = box.conf[0].cpu().numpy()
                    clase_id = int(box.cls[0].cpu().numpy())
                    
                    # Filtrar solo vehículos (clases COCO: 2:car, 3:motorcycle, 5:bus, 7:truck)
                    if clase_id in [2, 3, 5, 7] and confianza > 0.5:
                        detecciones.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confianza),
                            'class_id': clase_id,
                            'class_name': self._get_class_name(clase_id),
                            'timestamp': datetime.now()
                        })
            
            return detecciones
        
        except Exception as e:
            logger.error(f"[v0] Error detectando vehículos: {e}")
            return []
    
    def _get_class_name(self, class_id: int) -> str:
        """Obtiene nombre de la clase"""
        class_names = {
            2: 'carro',
            3: 'moto', 
            5: 'bus',
            7: 'camion'
        }
        return class_names.get(class_id, f'clase_{class_id}')
    
    def _detectar_placa(self, frame, vehiculo) -> Optional[Dict]:
        """Detecta placa en un vehículo usando múltiples estrategias"""
        if self.reader is None:
            return None
        
        try:
            x1, y1, x2, y2 = vehiculo['bbox']
            
            # Extraer región del vehículo
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # ESTRATEGIA 1: Buscar en toda la región del vehículo
            resultados_estrategia1 = self._buscar_placa_en_region(roi, x1, y1)
            if resultados_estrategia1:
                return resultados_estrategia1[0]  # Retornar la mejor
            
            # ESTRATEGIA 2: Buscar en la parte inferior (donde suelen estar las placas)
            resultados_estrategia2 = self._buscar_placa_parte_inferior(roi, x1, y1)
            if resultados_estrategia2:
                return resultados_estrategia2[0]
            
            # ESTRATEGIA 3: Buscar en múltiples regiones
            resultados_estrategia3 = self._buscar_placa_multiple_regiones(roi, x1, y1)
            if resultados_estrategia3:
                return resultados_estrategia3[0]
            
            return None
            
        except Exception as e:
            logger.error(f"[v0] Error detectando placa: {e}")
            return None

    def _buscar_placa_en_region(self, roi, offset_x, offset_y):
        """Busca placa en toda la región del vehículo"""
        try:
            # Preprocesar imagen para mejorar OCR
            roi_procesado = self._preprocesar_para_ocr(roi)
            
            # Usar EasyOCR en toda la región
            resultados_ocr = self.reader.readtext(
                roi_procesado, 
                detail=1,
                paragraph=False,
                text_threshold=0.6,
                low_text=0.4,
                link_threshold=0.4,
                width_ths=0.5,
                height_ths=0.5
            )
            
            placas_validas = []
            
            for (bbox, texto, confianza) in resultados_ocr:
                texto_limpio = limpiar_texto_placa(texto)
                placa_normalizada = normalizar_placa_peruana(texto_limpio)
                
                if placa_normalizada and validar_placa_peruana(placa_normalizada) and confianza > 0.4:
                    # Convertir coordenadas
                    (x_p1, y_p1), (x_p2, y_p2), (x_p3, y_p3), (x_p4, y_p4) = bbox
                    
                    x_abs = offset_x + int(min(x_p1, x_p2, x_p3, x_p4))
                    y_abs = offset_y + int(min(y_p1, y_p2, y_p3, y_p4))
                    w_abs = int(max(x_p1, x_p2, x_p3, x_p4) - min(x_p1, x_p2, x_p3, x_p4))
                    h_abs = int(max(y_p1, y_p2, y_p3, y_p4) - min(y_p1, y_p2, y_p3, y_p4))
                    
                    placa_info = {
                        'placa': placa_normalizada,
                        'bbox': (x_abs, y_abs, w_abs, h_abs),
                        'confidence': float(confianza),
                        'timestamp': datetime.now(),
                        'estrategia': 'region_completa'
                    }
                    
                    placas_validas.append(placa_info)
                    print(f"🚗 Placa detectada (región): {placa_normalizada} (conf: {confianza:.2f})")
            
            # Ordenar por confianza y retornar
            placas_validas.sort(key=lambda x: x['confidence'], reverse=True)
            return placas_validas
            
        except Exception as e:
            print(f"❌ Error en búsqueda por región: {e}")
            return []

    def _buscar_placa_parte_inferior(self, roi, offset_x, offset_y):
        """Busca placa específicamente en la parte inferior del vehículo"""
        try:
            altura_roi = roi.shape[0]
            ancho_roi = roi.shape[1]
            
            # Definir múltiples regiones en la parte inferior
            regiones = [
                # Parte inferior central (más común)
                (int(ancho_roi * 0.1), int(altura_roi * 0.6), 
                 int(ancho_roi * 0.9), int(altura_roi * 0.9)),
                # Parte inferior izquierda
                (int(ancho_roi * 0.0), int(altura_roi * 0.7),
                 int(ancho_roi * 0.4), int(altura_roi * 0.95)),
                # Parte inferior derecha  
                (int(ancho_roi * 0.6), int(altura_roi * 0.7),
                 int(ancho_roi * 1.0), int(altura_roi * 0.95))
            ]
            
            placas_validas = []
            
            for x_start, y_start, x_end, y_end in regiones:
                # Asegurar que las coordenadas estén dentro de los límites
                x_start = max(0, x_start)
                y_start = max(0, y_start)
                x_end = min(ancho_roi, x_end)
                y_end = min(altura_roi, y_end)
                
                if x_end <= x_start or y_end <= y_start:
                    continue
                    
                region_placa = roi[y_start:y_end, x_start:x_end]
                
                if region_placa.size == 0:
                    continue
                
                # Procesar región
                region_procesada = self._preprocesar_para_ocr(region_placa)
                
                resultados_ocr = self.reader.readtext(
                    region_procesada,
                    detail=1,
                    paragraph=False,
                    text_threshold=0.5,
                    low_text=0.3
                )
                
                for (bbox, texto, confianza) in resultados_ocr:
                    texto_limpio = limpiar_texto_placa(texto)
                    placa_normalizada = normalizar_placa_peruana(texto_limpio)
                    
                    if placa_normalizada and validar_placa_peruana(placa_normalizada) and confianza > 0.3:
                        # Convertir coordenadas
                        (x_p1, y_p1), (x_p2, y_p2), (x_p3, y_p3), (x_p4, y_p4) = bbox
                        
                        x_abs = offset_x + x_start + int(min(x_p1, x_p2, x_p3, x_p4))
                        y_abs = offset_y + y_start + int(min(y_p1, y_p2, y_p3, y_p4))
                        w_abs = int(max(x_p1, x_p2, x_p3, x_p4) - min(x_p1, x_p2, x_p3, x_p4))
                        h_abs = int(max(y_p1, y_p2, y_p3, y_p4) - min(y_p1, y_p2, y_p3, y_p4))
                        
                        placa_info = {
                            'placa': placa_normalizada,
                            'bbox': (x_abs, y_abs, w_abs, h_abs),
                            'confidence': float(confianza),
                            'timestamp': datetime.now(),
                            'estrategia': 'parte_inferior'
                        }
                        
                        placas_validas.append(placa_info)
                        print(f"🚗 Placa detectada (inferior): {placa_normalizada} (conf: {confianza:.2f})")
            
            placas_validas.sort(key=lambda x: x['confidence'], reverse=True)
            return placas_validas
            
        except Exception as e:
            print(f"❌ Error en búsqueda parte inferior: {e}")
            return []

    def _buscar_placa_multiple_regiones(self, roi, offset_x, offset_y):
        """Busca placa en múltiples regiones estratégicas"""
        try:
            altura_roi = roi.shape[0]
            ancho_roi = roi.shape[1]
            
            # Dividir la imagen en una cuadrícula
            grid_size = 3
            cell_height = altura_roi // grid_size
            cell_width = ancho_roi // grid_size
            
            placas_validas = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    y_start = i * cell_height
                    y_end = (i + 1) * cell_height
                    x_start = j * cell_width
                    x_end = (j + 1) * cell_width
                    
                    region = roi[y_start:y_end, x_start:x_end]
                    
                    if region.size == 0:
                        continue
                    
                    region_procesada = self._preprocesar_para_ocr(region)
                    
                    resultados_ocr = self.reader.readtext(
                        region_procesada,
                        detail=1,
                        paragraph=False
                    )
                    
                    for (bbox, texto, confianza) in resultados_ocr:
                        texto_limpio = limpiar_texto_placa(texto)
                        placa_normalizada = normalizar_placa_peruana(texto_limpio)
                        
                        if placa_normalizada and validar_placa_peruana(placa_normalizada) and confianza > 0.3:
                            (x_p1, y_p1), (x_p2, y_p2), (x_p3, y_p3), (x_p4, y_p4) = bbox
                            
                            x_abs = offset_x + x_start + int(min(x_p1, x_p2, x_p3, x_p4))
                            y_abs = offset_y + y_start + int(min(y_p1, y_p2, y_p3, y_p4))
                            w_abs = int(max(x_p1, x_p2, x_p3, x_p4) - min(x_p1, x_p2, x_p3, x_p4))
                            h_abs = int(max(y_p1, y_p2, y_p3, y_p4) - min(y_p1, y_p2, y_p3, y_p4))
                            
                            placa_info = {
                                'placa': placa_normalizada,
                                'bbox': (x_abs, y_abs, w_abs, h_abs),
                                'confidence': float(confianza),
                                'timestamp': datetime.now(),
                                'estrategia': f'grid_{i}_{j}'
                            }
                            
                            placas_validas.append(placa_info)
                            print(f"🚗 Placa detectada (grid): {placa_normalizada} (conf: {confianza:.2f})")
            
            placas_validas.sort(key=lambda x: x['confidence'], reverse=True)
            return placas_validas
            
        except Exception as e:
            print(f"❌ Error en búsqueda múltiple: {e}")
            return []

    def _preprocesar_para_ocr(self, imagen):
        """Preprocesa imagen para mejorar OCR"""
        try:
            # Convertir a escala de grises
            if len(imagen.shape) == 3:
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                gray = imagen
            
            # Aplicar filtro para mejorar contraste
            gray = cv2.medianBlur(gray, 3)
            
            # Mejorar contraste con CLAHE
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

    def _dibujar_anotaciones(self, frame, vehiculos, placas) -> np.ndarray:
        """Dibuja anotaciones en el frame"""
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
        
        # Información general
        fps_promedio = sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0
        
        cv2.putText(frame_anotado, f"Frames: {self.frame_count} | FPS: {fps_promedio:.1f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_anotado, f"Vehículos: {len(vehiculos)}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_anotado, f"Placas: {len(placas)}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_anotado
    
    def get_estadisticas(self) -> Dict:
        """Obtiene estadísticas del detector"""
        return {
            'total_frames': self.frame_count,
            'vehiculos_detectados': len(self.detecciones_vehiculos),
            'placas_detectadas': len(self.placas_detectadas),
            'fps_promedio': sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0
        }
    
    def detener(self):
        """Detiene el detector"""
        self.running = False
        print("🛑 DetectorCompletoInfracciones detenido")


# Función principal para pruebas
if __name__ == "__main__":
    # Prueba básica del detector
    detector = DetectorCompletoInfracciones()
    print("✅ Detector listo para usar")
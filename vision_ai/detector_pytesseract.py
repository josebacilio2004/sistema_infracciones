"""
Detector con pytesseract para OCR de placas - VERSIÓN MEJORADA PARA SERVIDOR
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
import os

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

# Configurar pytesseract con TU RUTA
try:
    import pytesseract
    # ✅ CONFIGURACIÓN PARA TU SISTEMA
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Bacilio\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    PYTESSERACT_DISPONIBLE = True
    print("✅ pytesseract configurado correctamente con tu ruta")
except ImportError:
    print("❌ pytesseract no disponible - ejecuta: pip install pytesseract")
    PYTESSERACT_DISPONIBLE = False
except Exception as e:
    print(f"⚠️ Error configurando pytesseract: {e}")
    PYTESSERACT_DISPONIBLE = False

logger = logging.getLogger(__name__)

class DetectorPytesseract:
    """Detector usando pytesseract para OCR - VERSIÓN MEJORADA PARA SERVIDOR"""
    
    def __init__(self, camara_id=None, **kwargs):
        print("🚀 Inicializando DetectorPytesseract MEJORADO...")
        
        self.camara_id = camara_id
        self.frame_count = 0
        self.detecciones_vehiculos = []
        self.placas_detectadas = []
        self.lock = threading.Lock()
        self.running = False
        
        # 🔥 MEJORA: OCR en CADA frame para máxima detección
        self.OCR_CADA_N_FRAMES = 1  # OCR cada frame
        
        # Métricas
        self.fps_real = deque(maxlen=30)
        self.tiempo_inicio = time.time()
        
        # Verificar Tesseract
        self._verificar_tesseract()
        
        # Cargar modelos
        self._cargar_modelos()
        
        print("✅ DetectorPytesseract MEJORADO inicializado correctamente")
    
    def _verificar_tesseract(self):
        """Verifica que Tesseract esté funcionando"""
        if not PYTESSERACT_DISPONIBLE:
            print("❌ pytesseract no disponible")
            return
        
        try:
            # Probar Tesseract
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version: {version}")
            
            # Probar con imagen de prueba simple
            test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            texto = pytesseract.image_to_string(test_image, config='--psm 8')
            if "TEST" in texto:
                print("✅ Tesseract funcionando correctamente")
            else:
                print("⚠️ Tesseract puede tener problemas de configuración")
                
        except Exception as e:
            print(f"❌ Error verificando Tesseract: {e}")
    
    def _cargar_modelos(self):
        """Carga YOLO y configura OCR"""
        try:
            # ✅ YOLOv8
            from ultralytics import YOLO
            print("📦 Cargando YOLOv8n...")
            self.modelo_yolo = YOLO('yolov8n.pt')
            self.modelo_yolo.fuse()
            print("✅ YOLOv8 cargado correctamente")
            
        except Exception as e:
            print(f"❌ Error cargando YOLO: {e}")
            self.modelo_yolo = None
    
    def procesar_frame(self, frame) -> Tuple[np.ndarray, List[Dict]]:
        """Procesa frame con pytesseract - VERSIÓN MEJORADA PARA SERVIDOR"""
        with self.lock:
            self.frame_count += 1
            
            # Calcular FPS
            tiempo_actual = time.time()
            self.fps_real.append(1 / (tiempo_actual - self.tiempo_inicio + 1e-6))
            self.tiempo_inicio = tiempo_actual
            
            # 1. Detectar vehículos (pero NO dependemos de esto)
            vehiculos = self._detectar_vehiculos_mejorado(frame)
            self.detecciones_vehiculos = vehiculos
            
            print(f"🔍 YOLO detectó {len(vehiculos)} vehículos")
            
            # 🔥 MEJORA PRINCIPAL: BUSCAR PLACAS SIEMPRE, incluso sin vehículos
            placas = []
            if PYTESSERACT_DISPONIBLE:
                # OCR en CADA frame para máxima sensibilidad
                if self.frame_count % self.OCR_CADA_N_FRAMES == 0:
                    print("🎯 Buscando placas en frame completo...")
                    placas = self._buscar_placas_completo_mejorado(frame, vehiculos)
            
            self.placas_detectadas = placas
            
            # 3. Dibujar anotaciones
            frame_anotado = self._dibujar_anotaciones(frame, vehiculos, placas)
            
            return frame_anotado, vehiculos
    
    def _detectar_vehiculos_mejorado(self, frame) -> List[Dict]:
        """Detección MEJORADA de vehículos con YOLO"""
        if self.modelo_yolo is None:
            return []
        
        try:
            detecciones = []
            
            # CONFIGURACIÓN MÁS PERMISIVA
            resultados = self.modelo_yolo(
                frame, 
                verbose=False,
                conf=0.25,  # ✅ Umbral MUCHO más bajo
                classes=[2, 3, 5, 7],  # Solo vehículos
                imgsz=640  # ✅ Tamaño fijo
            )
            
            for resultado in resultados:
                if resultado.boxes is not None:
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
            print(f"❌ Error detectando vehículos: {e}")
            return []
    
    def _buscar_placas_completo_mejorado(self, frame, vehiculos) -> List[Dict]:
        """Busca placas en TODO el frame - VERSIÓN MEJORADA"""
        placas_encontradas = []
        
        # 🔥 NUEVA ESTRATEGIA: Buscar en regiones estratégicas primero
        placas_regiones = self._buscar_en_regiones_estrategicas(frame)
        placas_encontradas.extend(placas_regiones)
        
        # ESTRATEGIA 2: Buscar en vehículos detectados (si hay)
        for vehiculo in vehiculos[:2]:
            placa = self._procesar_placa_pytesseract(frame, vehiculo)
            if placa:
                placas_encontradas.append(placa)
                print(f"🎯 PLACA ENCONTRADA en vehículo: {placa['placa']}")
        
        # ESTRATEGIA 3: OCR en frame completo (fallback final)
        if not placas_encontradas:
            print("🔍 No se encontraron placas, intentando OCR en frame completo...")
            placa_frame = self._buscar_placa_en_frame_completo(frame)
            if placa_frame:
                placas_encontradas.append(placa_frame)
                print(f"🎯 PLACA ENCONTRADA en frame completo: {placa_frame['placa']}")
        
        return placas_encontradas
    
    def _buscar_en_regiones_estrategicas(self, frame) -> List[Dict]:
        """Busca placas en regiones estratégicas del frame"""
        placas_encontradas = []
        
        try:
            altura, ancho = frame.shape[:2]
            
            # 🔥 REGIONES ESTRATÉGICAS donde suelen aparecer las placas
            regiones = [
                # Parte inferior central (placa delantera)
                (int(ancho * 0.2), int(altura * 0.7), int(ancho * 0.8), int(altura * 0.9)),
                # Parte superior central (placa trasera en algunos casos)
                (int(ancho * 0.2), int(altura * 0.1), int(ancho * 0.8), int(altura * 0.3)),
                # Lado izquierdo
                (int(ancho * 0.0), int(altura * 0.3), int(ancho * 0.3), int(altura * 0.7)),
                # Lado derecho  
                (int(ancho * 0.7), int(altura * 0.3), int(ancho * 1.0), int(altura * 0.7)),
            ]
            
            for i, (x1, y1, x2, y2) in enumerate(regiones):
                # Asegurar coordenadas válidas
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(ancho, x2), min(altura, y2)
                
                if x2 > x1 and y2 > y1:  # Validar región
                    region = frame[y1:y2, x1:x2]
                    
                    if region.size > 0:
                        texto = self._hacer_ocr_pytesseract(region)
                        if texto:
                            placas_encontradas.append({
                                'placa': texto,
                                'bbox': (x1, y1, x2 - x1, y2 - y1),
                                'confidence': 0.7,
                                'timestamp': datetime.now(),
                                'estrategia': f'region_estrategica_{i}'
                            })
                            print(f"🎯 Placa en región {i}: {texto}")
        
        except Exception as e:
            print(f"❌ Error en búsqueda por regiones: {e}")
        
        return placas_encontradas
    
    def _buscar_placa_en_frame_completo(self, frame) -> Optional[Dict]:
        """Busca placas en todo el frame"""
        try:
            # Procesar frame completo
            texto = self._hacer_ocr_pytesseract(frame)
            if texto:
                altura = frame.shape[0]
                ancho = frame.shape[1]
                return {
                    'placa': texto,
                    'bbox': (0, 0, ancho, altura),
                    'confidence': 0.6,
                    'timestamp': datetime.now(),
                    'estrategia': 'frame_completo'
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error buscando placa en frame completo: {e}")
            return None

    def _procesar_placa_pytesseract(self, frame, vehiculo) -> Optional[Dict]:
        """Procesa placa con pytesseract - Múltiples estrategias"""
        try:
            x1, y1, x2, y2 = vehiculo['bbox']
            roi_vehiculo = frame[y1:y2, x1:x2]
            
            if roi_vehiculo.size == 0:
                return None
            
            # ESTRATEGIA 1: Parte inferior del vehículo
            placa = self._buscar_en_parte_inferior(roi_vehiculo, x1, y1)
            if placa:
                return placa
            
            # ESTRATEGIA 2: Toda la región del vehículo
            placa = self._buscar_en_region_completa(roi_vehiculo, x1, y1)
            if placa:
                return placa
            
            return None
            
        except Exception as e:
            print(f"❌ Error procesando placa: {e}")
            return None
    
    def _buscar_en_parte_inferior(self, roi_vehiculo, offset_x, offset_y) -> Optional[Dict]:
        """Busca placa en la parte inferior del vehículo"""
        try:
            altura = roi_vehiculo.shape[0]
            ancho = roi_vehiculo.shape[1]
            
            # Múltiples regiones en la parte inferior
            regiones = [
                # Parte inferior central
                (int(ancho * 0.1), int(altura * 0.6), int(ancho * 0.9), int(altura * 0.9)),
                # Parte inferior completa
                (int(ancho * 0.0), int(altura * 0.7), int(ancho * 1.0), int(altura * 1.0)),
                # Parte delantera (para motos)
                (int(ancho * 0.3), int(altura * 0.5), int(ancho * 0.7), int(altura * 0.8))
            ]
            
            for x_start, y_start, x_end, y_end in regiones:
                region_placa = roi_vehiculo[y_start:y_end, x_start:x_end]
                
                if region_placa.size == 0:
                    continue
                
                # Procesar y hacer OCR
                texto = self._hacer_ocr_pytesseract(region_placa)
                if texto:
                    return {
                        'placa': texto,
                        'bbox': (offset_x + x_start, offset_y + y_start, x_end - x_start, y_end - y_start),
                        'confidence': 0.8,
                        'timestamp': datetime.now(),
                        'estrategia': 'parte_inferior'
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Error en búsqueda parte inferior: {e}")
            return None
    
    def _buscar_en_region_completa(self, roi_vehiculo, offset_x, offset_y) -> Optional[Dict]:
        """Busca placa en toda la región del vehículo"""
        try:
            texto = self._hacer_ocr_pytesseract(roi_vehiculo)
            if texto:
                altura = roi_vehiculo.shape[0]
                ancho = roi_vehiculo.shape[1]
                return {
                    'placa': texto,
                    'bbox': (offset_x, offset_y, ancho, altura),
                    'confidence': 0.7,
                    'timestamp': datetime.now(),
                    'estrategia': 'region_completa'
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error en búsqueda región completa: {e}")
            return None
    
    def _hacer_ocr_pytesseract(self, imagen) -> Optional[str]:
        """Ejecuta OCR con pytesseract y múltiples configuraciones"""
        if not PYTESSERACT_DISPONIBLE:
            return None
        
        try:
            # Preprocesar imagen
            imagen_procesada = self._preprocesar_para_ocr(imagen)
            
            # 🔥 MEJORA: Configuraciones más agresivas para detección
            configuraciones = [
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Palabra única
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Línea única
                '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', # Línea con segmentación
                '--psm 6',  # Bloque uniforme de texto
                '--psm 11',  # Texto denso
            ]
            
            for config in configuraciones:
                try:
                    texto = pytesseract.image_to_string(imagen_procesada, config=config)
                    texto_limpio = limpiar_texto_placa(texto)
                    
                    # 🔥 MEJORA: Validación menos estricta para testing
                    if texto_limpio and len(texto_limpio) >= 4:  # Mínimo 4 caracteres
                        # Intentar normalizar
                        placa_normalizada = normalizar_placa_peruana(texto_limpio)
                        if placa_normalizada:
                            print(f"🔍 OCR exitoso: '{texto_limpio}' -> '{placa_normalizada}'")
                            return placa_normalizada
                        else:
                            # Si no se puede normalizar, devolver el texto limpio
                            print(f"🔍 OCR detectó texto (no normalizado): '{texto_limpio}'")
                            return texto_limpio
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            print(f"❌ Error en OCR: {e}")
            return None
    
    def _preprocesar_para_ocr(self, imagen):
        """Preprocesamiento avanzado para OCR"""
        try:
            # Convertir a escala de grises
            if len(imagen.shape) == 3:
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                gray = imagen
            
            # 🔥 MEJORA: Probar múltiples técnicas y elegir la mejor
            procesadas = []
            
            # 1. Original + CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_clahe = clahe.apply(gray)
            procesadas.append(gray_clahe)
            
            # 2. Suavizado + umbral
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            procesadas.append(thresh)
            
            # 3. Morfología para mejorar texto
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            procesadas.append(morph)
            
            # 4. Bilateral filter para preservar bordes
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            procesadas.append(bilateral)
            
            # Probar con la primera imagen procesada (puedes cambiar esto)
            return procesadas[0]
            
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
    
    def _dibujar_anotaciones(self, frame, vehiculos, placas) -> np.ndarray:
        """Dibuja anotaciones en el frame"""
        frame_anotado = frame.copy()
        
        # Dibujar vehículos
        for vehiculo in vehiculos:
            x1, y1, x2, y2 = vehiculo['bbox']
            color = (0, 255, 0)  # Verde para vehículos
            
            cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color, 2)
            
            etiqueta = f"{vehiculo['class_name']} ({vehiculo['confidence']:.2f})"
            cv2.putText(frame_anotado, etiqueta,
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Dibujar placas (en rojo)
        for placa in placas:
            x, y, w, h = placa['bbox']
            cv2.rectangle(frame_anotado, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            etiqueta_placa = f"PLACA: {placa['placa']}"
            cv2.putText(frame_anotado, etiqueta_placa,
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Información del sistema
        fps_promedio = sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0
        
        # Panel de información MEJORADO
        cv2.rectangle(frame_anotado, (5, 5), (450, 120), (0, 0, 0), -1)
        cv2.rectangle(frame_anotado, (5, 5), (450, 120), (0, 255, 0), 2)
        
        cv2.putText(frame_anotado, f"FPS: {fps_promedio:.1f}", 
                   (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_anotado, f"Vehículos: {len(vehiculos)}", 
                   (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_anotado, f"Placas: {len(placas)}", 
                   (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_anotado, f"Frame: {self.frame_count}", 
                   (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame_anotado, "OCR: ACTIVO", 
                   (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Mostrar última placa detectada
        if placas:
            ultima_placa = placas[-1]['placa']
            cv2.putText(frame_anotado, f"Ultima: {ultima_placa}", 
                       (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame_anotado
    
    def get_estadisticas(self) -> Dict:
        """Obtiene estadísticas MEJORADAS"""
        stats = {
            'total_frames': self.frame_count,
            'vehiculos_detectados': len(self.detecciones_vehiculos),
            'placas_detectadas': len(self.placas_detectadas),
            'fps_promedio': sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0,
            'detector': 'pytesseract_mejorado'
        }
        
        # Agregar información de las últimas placas
        if self.placas_detectadas:
            stats['ultimas_placas'] = []
            for placa in self.placas_detectadas[-3:]:  # Últimas 3 placas
                stats['ultimas_placas'].append({
                    'placa': placa.get('placa', 'N/A'),
                    'confianza': placa.get('confidence', 0),
                    'estrategia': placa.get('estrategia', 'desconocida')
                })
        
        return stats
    
    def detener(self):
        """Detiene el detector"""
        self.running = False
        print("🛑 DetectorPytesseract MEJORADO detenido")
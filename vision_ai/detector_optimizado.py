"""
Sistema de detección OPTIMIZADO para máximo rendimiento (FPS)
Enfocado en 3 infracciones: Luz Roja, Exceso de Velocidad, Invasión de Carril
Optimizaciones: Skip frames, resolución reducida, OCR threading, GPU acceleration
"""
import os
import sys
import django
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import re
import threading
from queue import Queue
from collections import deque

# Configurar Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

from ultralytics import YOLO
from infracciones.models import Infraccion, Vehiculo, TipoInfraccion, EventoDeteccion
from camaras.models import Camara
from ml_predicciones.predictor import PredictorRiesgo

try:
    import easyocr
    OCR_DISPONIBLE = True
except ImportError:
    OCR_DISPONIBLE = False
    print("⚠️  EasyOCR no disponible, detección de placas deshabilitada")


class DetectorOptimizado:
    """Detector de infracciones OPTIMIZADO para máximo FPS"""
    
    def __init__(self, camara_id=0, usar_gpu=True):
        print("🚀 Inicializando detector OPTIMIZADO...")
        
        self.SKIP_FRAMES = 2  # Procesar 1 de cada 3 frames (3x más rápido)
        self.RESOLUCION_PROCESAMIENTO = (640, 480)  # Resolución reducida para procesamiento
        self.RESOLUCION_DISPLAY = (1280, 720)  # Resolución para mostrar
        self.CONFIANZA_MIN = 0.5  # Umbral de confianza
        self.OCR_CADA_N_FRAMES = 60  # Ejecutar OCR cada 60 frames (2 segundos a 30fps)
        
        self.modelo_yolo = YOLO('yolov8n.pt')  # Modelo nano (más rápido)
        self.modelo_yolo.fuse()  # Fusionar capas para mayor velocidad
        
        # Configurar para GPU si está disponible
        if usar_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("✅ GPU detectada, usando aceleración CUDA")
            self.usar_gpu = True
        else:
            print("⚠️  GPU no disponible, usando CPU")
            self.usar_gpu = False
        
        print("✅ Modelo YOLO nano cargado y optimizado")
        
        self.ocr_queue = Queue(maxsize=5)
        self.ocr_results = {}
        self.ocr_activo = False
        
        if OCR_DISPONIBLE:
            self.reader = easyocr.Reader(['en'], gpu=self.usar_gpu)
            self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
            self.ocr_thread.start()
            self.ocr_activo = True
            print("✅ OCR inicializado en thread separado")
        
        # Inicializar predictor ML
        self.predictor_ml = PredictorRiesgo()
        print("✅ Predictor ML inicializado")
        
        self.cap = cv2.VideoCapture(camara_id)
        if not self.cap.isOpened():
            raise Exception("❌ No se pudo abrir la webcam")
        
        # Configurar para máximo rendimiento
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.RESOLUCION_DISPLAY[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.RESOLUCION_DISPLAY[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para reducir latencia
        
        print(f"✅ Webcam configurada ({self.RESOLUCION_DISPLAY[0]}x{self.RESOLUCION_DISPLAY[1]})")
        
        # Obtener o crear cámara en BD
        self.camara_db, created = Camara.objects.get_or_create(
            ubicacion="Webcam Local - Optimizado",
            defaults={
                'ip': '127.0.0.1', 
                'descripcion': 'Cámara optimizada para alto rendimiento',
                'activa': True
            }
        )
        
        self.fps = 30
        self.frame_count = 0
        self.fps_real = deque(maxlen=30)  # Calcular FPS real
        self.ultimo_tiempo = cv2.getTickCount()
        
        # Tracking de vehículos
        self.vehiculos_trackeados = {}
        self.placas_detectadas = {}
        
        self.LIMITE_VELOCIDAD = 60  # km/h
        self.DISTANCIA_METROS = 20
        self.MARGEN_CARRIL = 50  # píxeles
        
        # Cooldown para evitar detecciones duplicadas
        self.cooldown_infracciones = {}  # {vehiculo_id: {tipo: timestamp}}
        self.COOLDOWN_SEGUNDOS = 5
        
        # Crear carpetas
        self.carpeta_evidencias = BASE_DIR / 'media' / 'infracciones' / 'imagenes'
        self.carpeta_placas = BASE_DIR / 'media' / 'infracciones' / 'placas'
        self.carpeta_evidencias.mkdir(parents=True, exist_ok=True)
        self.carpeta_placas.mkdir(parents=True, exist_ok=True)
        
        print("✅ Sistema OPTIMIZADO listo\n")
        print(f"⚡ Configuración: Skip {self.SKIP_FRAMES} frames, "
              f"Resolución {self.RESOLUCION_PROCESAMIENTO}, "
              f"OCR cada {self.OCR_CADA_N_FRAMES} frames")
    
    def _ocr_worker(self):
        """Worker thread para procesar OCR en paralelo"""
        while True:
            try:
                vehiculo_id, roi = self.ocr_queue.get()
                if roi is None:
                    break
                
                placa, confianza = self._detectar_placa_ocr(roi)
                if placa:
                    self.ocr_results[vehiculo_id] = (placa, confianza)
                
                self.ocr_queue.task_done()
            except Exception as e:
                print(f"⚠️  Error en OCR worker: {e}")
    
    def _detectar_placa_ocr(self, roi):
        """Detecta placa usando OCR (ejecutado en thread separado)"""
        try:
            if roi.size == 0:
                return None, 0
            
            # Preprocesar
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            # OCR
            resultados = self.reader.readtext(thresh, detail=1)
            
            for (bbox, texto, confianza) in resultados:
                if confianza > 0.5:
                    placa_limpia = self._limpiar_placa(texto)
                    if placa_limpia:
                        return placa_limpia, confianza
            
            return None, 0
        except Exception as e:
            return None, 0
    
    def _limpiar_placa(self, texto):
        """Limpia y valida texto de placa"""
        texto = re.sub(r'[^A-Z0-9]', '', texto.upper())
        if 6 <= len(texto) <= 8:
            return texto
        return None
    
    def _puede_registrar_infraccion(self, vehiculo_id, tipo_infraccion):
        """Verifica si puede registrar infracción (cooldown)"""
        ahora = datetime.now()
        
        if vehiculo_id not in self.cooldown_infracciones:
            self.cooldown_infracciones[vehiculo_id] = {}
        
        if tipo_infraccion in self.cooldown_infracciones[vehiculo_id]:
            ultimo = self.cooldown_infracciones[vehiculo_id][tipo_infraccion]
            if (ahora - ultimo).total_seconds() < self.COOLDOWN_SEGUNDOS:
                return False
        
        self.cooldown_infracciones[vehiculo_id][tipo_infraccion] = ahora
        return True
    
    def detectar_luz_roja(self, frame, resultados):
        """Detecta semáforo en rojo (OPTIMIZADO)"""
        for box in resultados[0].boxes:
            cls = self.modelo_yolo.names[int(box.cls)]
            
            if cls == 'traffic light':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                semaforo = frame[y1:y2, x1:x2]
                
                if semaforo.size == 0:
                    continue
                
                hsv = cv2.cvtColor(semaforo, cv2.COLOR_BGR2HSV)
                
                # Máscaras para rojo
                mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                mascara_roja = cv2.bitwise_or(mask1, mask2)
                
                if cv2.countNonZero(mascara_roja) > 50:
                    return True, (x1, y1, x2, y2)
        
        return False, None
    
    def detectar_exceso_velocidad(self, vehiculo_id, posicion_actual):
        """Detecta exceso de velocidad (OPTIMIZADO)"""
        if vehiculo_id in self.vehiculos_trackeados:
            datos_anteriores = self.vehiculos_trackeados[vehiculo_id]
            frame_diff = self.frame_count - datos_anteriores['frame']
            
            # Calcular distancia euclidiana
            pos_anterior = datos_anteriores['posicion']
            distancia_pixeles = np.linalg.norm(
                np.array(posicion_actual) - np.array(pos_anterior)
            )
            
            # Convertir a velocidad (aproximación)
            tiempo_segundos = frame_diff / self.fps
            if tiempo_segundos > 0 and distancia_pixeles > 10:
                # Factor de conversión píxeles a metros (ajustar según calibración)
                factor_conversion = 0.05
                distancia_metros = distancia_pixeles * factor_conversion
                velocidad = (distancia_metros / tiempo_segundos) * 3.6
                
                if self.LIMITE_VELOCIDAD < velocidad < 200:
                    return True, velocidad
        
        return False, 0
    
    def detectar_invasion_carril(self, frame, x1, y1, x2, y2):
        """Detecta invasión de carril (OPTIMIZADO)"""
        h, w = frame.shape[:2]
        centro_x = (x1 + x2) // 2
        
        # Línea central del frame
        linea_central = w // 2
        
        # Si el vehículo está muy cerca de la línea central
        if abs(centro_x - linea_central) < self.MARGEN_CARRIL:
            return True
        
        return False
    
    def registrar_infraccion(self, tipo_codigo, frame, vehiculo_placa="DESCONOCIDA", 
                            velocidad=None, confianza=0.85):
        """Registra infracción en BD (OPTIMIZADO - async)"""
        try:
            vehiculo, _ = Vehiculo.objects.get_or_create(
                placa=vehiculo_placa,
                defaults={'tipo_vehiculo': 'AUTO'}
            )
            
            tipo_infraccion = TipoInfraccion.objects.filter(codigo=tipo_codigo).first()
            if not tipo_infraccion:
                return None
            
            # Guardar imagen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"{tipo_codigo}_{vehiculo_placa}_{timestamp}.jpg"
            ruta_imagen = self.carpeta_evidencias / nombre_archivo
            
            threading.Thread(
                target=cv2.imwrite, 
                args=(str(ruta_imagen), frame),
                daemon=True
            ).start()
            
            # Crear infracción
            infraccion = Infraccion.objects.create(
                vehiculo=vehiculo,
                tipo_infraccion=tipo_infraccion,
                camara=self.camara_db,
                ubicacion=self.camara_db.ubicacion,
                velocidad_detectada=int(velocidad) if velocidad else None,
                velocidad_maxima=self.LIMITE_VELOCIDAD if velocidad else None,
                imagen_principal=f'infracciones/imagenes/{nombre_archivo}',
                confianza_deteccion=confianza * 100,
                modelo_ia_version='YOLOv8n-Optimizado',
                estado='DETECTADA'
            )
            
            print(f"✅ {tipo_infraccion.nombre} - {vehiculo_placa}")
            return infraccion
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def procesar_frame(self, frame):
        """Procesa frame (OPTIMIZADO)"""
        self.frame_count += 1
        
        if self.frame_count % (self.SKIP_FRAMES + 1) != 0:
            return frame
        
        frame_small = cv2.resize(frame, self.RESOLUCION_PROCESAMIENTO)
        
        # Ejecutar YOLO
        resultados = self.modelo_yolo.track(
            frame_small, 
            persist=True, 
            verbose=False,
            conf=self.CONFIANZA_MIN,
            iou=0.5
        )
        
        if not resultados or len(resultados[0].boxes) == 0:
            return frame
        
        # Detectar luz roja
        luz_roja, coords_semaforo = self.detectar_luz_roja(frame_small, resultados)
        
        scale_x = frame.shape[1] / self.RESOLUCION_PROCESAMIENTO[0]
        scale_y = frame.shape[0] / self.RESOLUCION_PROCESAMIENTO[1]
        
        # Procesar vehículos
        for box in resultados[0].boxes:
            cls = self.modelo_yolo.names[int(box.cls)]
            conf = float(box.conf[0])
            
            if cls not in ['car', 'truck', 'bus', 'motorcycle']:
                continue
            
            # Escalar coordenadas al tamaño original
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            vehiculo_id = int(box.id[0]) if box.id is not None else None
            
            if vehiculo_id:
                centro = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                if (self.ocr_activo and 
                    self.frame_count % self.OCR_CADA_N_FRAMES == 0 and
                    vehiculo_id not in self.placas_detectadas and
                    not self.ocr_queue.full()):
                    
                    roi = frame[y1:y2, x1:x2].copy()
                    self.ocr_queue.put((vehiculo_id, roi))
                
                # Obtener placa
                if vehiculo_id in self.ocr_results:
                    placa, _ = self.ocr_results[vehiculo_id]
                    self.placas_detectadas[vehiculo_id] = placa
                
                placa_vehiculo = self.placas_detectadas.get(vehiculo_id, f"VEH-{vehiculo_id:04d}")
                
                # Detectar exceso de velocidad
                exceso, velocidad = self.detectar_exceso_velocidad(vehiculo_id, centro)
                
                if exceso and self._puede_registrar_infraccion(vehiculo_id, 'EXCESO_VEL'):
                    self.registrar_infraccion('EXCESO_VEL', frame, placa_vehiculo, 
                                            velocidad=velocidad, confianza=conf)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"EXCESO: {velocidad:.0f} km/h", 
                              (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, placa_vehiculo, 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # Actualizar tracking
                    self.vehiculos_trackeados[vehiculo_id] = {
                        'frame': self.frame_count,
                        'posicion': centro,
                        'placa': placa_vehiculo
                    }
                    
                    # Detectar invasión de carril
                    invasion = self.detectar_invasion_carril(frame, x1, y1, x2, y2)
                    
                    if invasion and self._puede_registrar_infraccion(vehiculo_id, 'INVASION_CARRIL'):
                        self.registrar_infraccion('INVASION_CARRIL', frame, placa_vehiculo, 
                                                confianza=conf)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                        cv2.putText(frame, "INVASION CARRIL", 
                                  (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.putText(frame, placa_vehiculo, 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{cls} {conf:.2f}", 
                                  (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, placa_vehiculo, 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Luz roja
                if luz_roja and self._puede_registrar_infraccion(vehiculo_id, 'LUZ_ROJA'):
                    self.registrar_infraccion('LUZ_ROJA', frame, placa_vehiculo, confianza=conf)
                    cv2.putText(frame, "LUZ ROJA!", 
                              (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        tiempo_actual = cv2.getTickCount()
        tiempo_transcurrido = (tiempo_actual - self.ultimo_tiempo) / cv2.getTickFrequency()
        fps_actual = 1.0 / tiempo_transcurrido if tiempo_transcurrido > 0 else 0
        self.fps_real.append(fps_actual)
        self.ultimo_tiempo = tiempo_actual
        
        fps_promedio = sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0
        
        # Dibujar info
        cv2.rectangle(frame, (5, 5), (450, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (450, 120), (0, 255, 0), 2)
        
        cv2.putText(frame, f"FPS: {fps_promedio:.1f} | Frame: {self.frame_count}", 
                   (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Vehiculos: {len(self.vehiculos_trackeados)}", 
                   (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Placas: {len(self.placas_detectadas)}", 
                   (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Modelo: YOLOv8n-Optimizado", 
                   (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if luz_roja and coords_semaforo:
            x1, y1, x2, y2 = coords_semaforo
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "SEMAFORO ROJO", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def iniciar_deteccion(self):
        """Inicia detección en tiempo real"""
        print("\n🎥 Iniciando detección OPTIMIZADA...")
        print("Presiona 'q' para salir\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Error al capturar frame")
                    break
                
                frame_procesado = self.procesar_frame(frame)
                
                cv2.imshow('Sistema OPTIMIZADO - Tesis (Luz Roja | Velocidad | Carril)', 
                          frame_procesado)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrumpido por usuario")
        finally:
            self.detener()
    
    def detener(self):
        """Libera recursos"""
        print("\n🛑 Deteniendo sistema...")
        
        if self.ocr_activo:
            self.ocr_queue.put((None, None))
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Sistema detenido")


def main():
    print("=" * 70)
    print("🚦 SISTEMA OPTIMIZADO DE DETECCIÓN DE INFRACCIONES")
    print("📚 Tesis - IoT + Visión AI + ML + OCR")
    print("⚡ Optimizado para máximo rendimiento (FPS)")
    print("=" * 70)
    print()
    
    try:
        detector = DetectorOptimizado(camara_id=0, usar_gpu=True)
        detector.iniciar_deteccion()
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

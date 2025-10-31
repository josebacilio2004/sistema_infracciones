"""
Detector optimizado para placas peruanas con YOLOv8 + EasyOCR
Formato de placa peruana: A1B-234 (3 alfanuméricos + guión + 3 números)
Optimizado para máximo rendimiento en FPS
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
from collections import deque

# Configurar Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

from ultralytics import YOLO
import easyocr
from infracciones.models import Infraccion, Vehiculo, TipoInfraccion, EventoDeteccion
from camaras.models import Camara
from ml_predicciones.predictor import PredictorRiesgo

class DetectorPlacasPeru:
    """Detector optimizado para placas peruanas con alto rendimiento"""
    
    def __init__(self, camara_id=0, skip_frames=2, usar_gpu=True):
        print("🚀 Inicializando detector optimizado para placas peruanas...")
        
        self.skip_frames = skip_frames  # Procesar 1 de cada N frames
        self.frame_count = 0
        self.fps_real = deque(maxlen=30)
        self.ultimo_tiempo = datetime.now()
        
        self.modelo_yolo = YOLO('yolov8n.pt')
        self.modelo_yolo.fuse()  # Fusionar capas para mayor velocidad
        print("✅ YOLOv8n cargado y fusionado")
        
        print("📝 Cargando OCR optimizado para placas peruanas...")
        self.reader = easyocr.Reader(
            ['en'], 
            gpu=usar_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0,
            model_storage_directory=str(BASE_DIR / 'models' / 'easyocr'),
            download_enabled=True,
            verbose=False
        )
        print("✅ OCR inicializado")
        
        self.predictor_ml = PredictorRiesgo()
        
        self.cap = cv2.VideoCapture(camara_id)
        if not self.cap.isOpened():
            raise Exception("❌ No se pudo abrir la webcam")
        
        # Resolución reducida para mejor rendimiento
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
        print("✅ Webcam configurada (1280x720@30fps)")
        
        self.camara_db, _ = Camara.objects.get_or_create(
            ubicacion="Webcam Local - Detección Placas Perú",
            defaults={
                'ip': '127.0.0.1',
                'descripcion': 'Detector optimizado con OCR para placas peruanas',
                'activa': True
            }
        )
        
        self.vehiculos_trackeados = {}
        self.placas_detectadas = {}
        self.cooldown_infracciones = {}  # Evitar duplicados
        self.cooldown_tiempo = 5  # segundos
        
        self.LIMITE_VELOCIDAD = 60  # km/h
        self.DISTANCIA_METROS = 20
        self.fps_camara = 30
        
        self.ocr_queue = []
        self.ocr_results = {}
        self.ocr_thread = None
        self.ocr_running = False
        
        self.carpeta_evidencias = BASE_DIR / 'media' / 'infracciones' / 'imagenes'
        self.carpeta_placas = BASE_DIR / 'media' / 'infracciones' / 'placas'
        self.carpeta_evidencias.mkdir(parents=True, exist_ok=True)
        self.carpeta_placas.mkdir(parents=True, exist_ok=True)
        
        print("✅ Sistema listo - Optimizado para placas peruanas (A1B-234)\n")
    
    def validar_placa_peruana(self, texto):
        """
        Valida y limpia placas peruanas
        Formato: A1B-234 (3 alfanuméricos + guión + 3 números)
        """
        # Limpiar texto
        texto = re.sub(r'[^A-Z0-9-]', '', texto.upper())
        
        # Patrón para placas peruanas: 3 alfanuméricos + guión + 3 números
        patron_peru = r'^[A-Z0-9]{3}-?[0-9]{3}$'
        
        if re.match(patron_peru, texto):
            # Asegurar que tenga el guión
            if '-' not in texto and len(texto) == 6:
                texto = texto[:3] + '-' + texto[3:]
            return texto
        
        # Intentar corregir errores comunes del OCR
        if len(texto) >= 6:
            # Remover caracteres extra
            texto = texto[:7] if '-' in texto else texto[:6]
            
            # Agregar guión si falta
            if '-' not in texto and len(texto) == 6:
                texto = texto[:3] + '-' + texto[3:]
            
            # Validar nuevamente
            if re.match(patron_peru, texto):
                return texto
        
        return None
    
    def detectar_placa_optimizada(self, frame, x1, y1, x2, y2, vehiculo_id):
        """
        Detecta placa con preprocesamiento optimizado para placas peruanas
        (fondo blanco, texto negro)
        """
        try:
            h, w = frame.shape[:2]
            margen = 15
            y1_exp = max(0, y1 - margen)
            y2_exp = min(h, y2 + margen)
            x1_exp = max(0, x1 - margen)
            x2_exp = min(w, x2 + margen)
            
            roi = frame[y1_exp:y2_exp, x1_exp:x2_exp]
            
            if roi.size == 0:
                return None, None, None
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Aumentar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Filtro bilateral para reducir ruido manteniendo bordes
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Threshold adaptativo para placas blancas
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Invertir si es necesario (texto debe ser blanco sobre negro para OCR)
            if np.mean(thresh) > 127:
                thresh = cv2.bitwise_not(thresh)
            
            resultados = self.reader.readtext(
                thresh,
                detail=1,
                paragraph=False,
                min_size=10,
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.3,
                canvas_size=2560,
                mag_ratio=1.5
            )
            
            mejor_placa = None
            mejor_confianza = 0
            mejor_roi = None
            
            for (bbox, texto, confianza) in resultados:
                if confianza > 0.4:  # Umbral más bajo para capturar más candidatos
                    placa_validada = self.validar_placa_peruana(texto)
                    if placa_validada and confianza > mejor_confianza:
                        mejor_placa = placa_validada
                        mejor_confianza = confianza
                        mejor_roi = roi.copy()
            
            return mejor_placa, mejor_confianza, mejor_roi
            
        except Exception as e:
            print(f"⚠️  Error en OCR: {e}")
            return None, None, None
    
    def calcular_fps(self):
        """Calcula FPS real del sistema"""
        ahora = datetime.now()
        delta = (ahora - self.ultimo_tiempo).total_seconds()
        if delta > 0:
            fps = 1.0 / delta
            self.fps_real.append(fps)
        self.ultimo_tiempo = ahora
        return sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0
    
    def puede_registrar_infraccion(self, vehiculo_id, tipo_codigo):
        """Verifica si se puede registrar una infracción (cooldown)"""
        clave = f"{vehiculo_id}_{tipo_codigo}"
        ahora = datetime.now()
        
        if clave in self.cooldown_infracciones:
            ultimo_registro = self.cooldown_infracciones[clave]
            if (ahora - ultimo_registro).total_seconds() < self.cooldown_tiempo:
                return False
        
        self.cooldown_infracciones[clave] = ahora
        return True
    
    def detectar_luz_roja(self, frame, resultados):
        """Detecta semáforo en rojo"""
        for box in resultados[0].boxes:
            cls = self.modelo_yolo.names[int(box.cls)]
            
            if cls == 'traffic light':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                semaforo = frame[y1:y2, x1:x2]
                
                if semaforo.size == 0:
                    continue
                
                # Detectar color rojo en HSV
                hsv = cv2.cvtColor(semaforo, cv2.COLOR_BGR2HSV)
                
                # Rangos para rojo
                rojo_bajo1 = np.array([0, 120, 70])
                rojo_alto1 = np.array([10, 255, 255])
                rojo_bajo2 = np.array([170, 120, 70])
                rojo_alto2 = np.array([180, 255, 255])
                
                mascara1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
                mascara2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
                mascara_roja = cv2.bitwise_or(mascara1, mascara2)
                
                pixeles_rojos = cv2.countNonZero(mascara_roja)
                
                if pixeles_rojos > 30:
                    return True, (x1, y1, x2, y2)
        
        return False, None
    
    def detectar_exceso_velocidad(self, vehiculo_id, frame_actual):
        """Detecta exceso de velocidad"""
        if vehiculo_id in self.vehiculos_trackeados:
            frame_anterior = self.vehiculos_trackeados[vehiculo_id]['frame']
            frame_diff = frame_actual - frame_anterior
            
            tiempo_segundos = frame_diff / self.fps_camara
            
            if tiempo_segundos > 0:
                velocidad = (self.DISTANCIA_METROS / tiempo_segundos) * 3.6
                
                if velocidad > self.LIMITE_VELOCIDAD and velocidad < 200:
                    return True, velocidad
        
        return False, 0
    
    def detectar_invasion_carril(self, frame, x1, y1, x2, y2):
        """Detecta invasión de carril"""
        centro_x = (x1 + x2) // 2
        h, w = frame.shape[:2]
        linea_central = w // 2
        margen_carril = w // 6
        
        if abs(centro_x - linea_central) < margen_carril // 2:
            return True
        
        return False
    
    def registrar_infraccion_async(self, tipo_codigo, frame, vehiculo_placa, 
                                   velocidad=None, confianza=0.85, imagen_placa=None):
        """Registra infracción de forma asíncrona"""
        def guardar():
            try:
                vehiculo, _ = Vehiculo.objects.get_or_create(
                    placa=vehiculo_placa,
                    defaults={'tipo_vehiculo': 'AUTO'}
                )
                
                tipo_infraccion = TipoInfraccion.objects.filter(codigo=tipo_codigo).first()
                if not tipo_infraccion:
                    return
                
                # Guardar imágenes
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"{tipo_codigo}_{vehiculo_placa}_{timestamp}.jpg"
                ruta_imagen = self.carpeta_evidencias / nombre_archivo
                cv2.imwrite(str(ruta_imagen), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                ruta_placa_rel = None
                if imagen_placa is not None:
                    nombre_placa = f"placa_{vehiculo_placa}_{timestamp}.jpg"
                    ruta_placa = self.carpeta_placas / nombre_placa
                    cv2.imwrite(str(ruta_placa), imagen_placa, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    ruta_placa_rel = f'infracciones/placas/{nombre_placa}'
                
                # Crear infracción
                Infraccion.objects.create(
                    vehiculo=vehiculo,
                    tipo_infraccion=tipo_infraccion,
                    camara=self.camara_db,
                    ubicacion=self.camara_db.ubicacion,
                    velocidad_detectada=int(velocidad) if velocidad else None,
                    velocidad_maxima=self.LIMITE_VELOCIDAD if velocidad else None,
                    imagen_principal=f'infracciones/imagenes/{nombre_archivo}',
                    imagen_placa=ruta_placa_rel,
                    confianza_deteccion=confianza * 100,
                    modelo_ia_version='YOLOv8n + EasyOCR (Placas Perú)',
                    estado='DETECTADA'
                )
                
                EventoDeteccion.objects.create(
                    camara=self.camara_db,
                    tipo_evento='INFRACCION_DETECTADA',
                    datos_evento={
                        'tipo': tipo_codigo,
                        'placa': vehiculo_placa,
                        'velocidad': velocidad,
                        'confianza': confianza,
                        'formato_placa': 'PERU'
                    }
                )
                
                print(f"✅ Infracción registrada: {tipo_infraccion.nombre} - {vehiculo_placa}")
                
            except Exception as e:
                print(f"❌ Error al registrar infracción: {e}")
        
        # Ejecutar en thread separado
        thread = threading.Thread(target=guardar, daemon=True)
        thread.start()
    
    def procesar_frame(self, frame):
        """Procesa frame con optimizaciones de rendimiento"""
        self.frame_count += 1
        
        if self.frame_count % (self.skip_frames + 1) != 0:
            return frame
        
        frame_display = frame.copy()
        fps_actual = self.calcular_fps()
        
        escala = 0.75
        frame_small = cv2.resize(frame, None, fx=escala, fy=escala)
        
        resultados = self.modelo_yolo.track(
            frame_small,
            persist=True,
            verbose=False,
            conf=0.4,
            iou=0.5,
            classes=[2, 3, 5, 7]  # car, motorcycle, bus, truck
        )
        
        if not resultados or len(resultados[0].boxes) == 0:
            self.dibujar_info_sistema(frame_display, fps_actual)
            return frame_display
        
        # Detectar luz roja
        luz_roja, coords_semaforo = self.detectar_luz_roja(frame_small, resultados)
        
        # Procesar vehículos
        for box in resultados[0].boxes:
            cls = self.modelo_yolo.names[int(box.cls)]
            conf = float(box.conf[0])
            
            if cls not in ['car', 'truck', 'bus', 'motorcycle']:
                continue
            
            # Escalar coordenadas de vuelta
            x1, y1, x2, y2 = map(int, box.xyxy[0] / escala)
            
            vehiculo_id = int(box.id[0]) if box.id is not None else None
            
            if vehiculo_id:
                placa_detectada = None
                confianza_placa = 0
                roi_placa = None
                
                if self.frame_count % 20 == 0 or vehiculo_id not in self.placas_detectadas:
                    placa_detectada, confianza_placa, roi_placa = self.detectar_placa_optimizada(
                        frame, x1, y1, x2, y2, vehiculo_id
                    )
                    if placa_detectada:
                        self.placas_detectadas[vehiculo_id] = placa_detectada
                        print(f"🚗 Placa peruana detectada: {placa_detectada} (conf: {confianza_placa:.2f})")
                
                placa_vehiculo = self.placas_detectadas.get(vehiculo_id, f"VEH-{vehiculo_id:04d}")
                
                # Detectar infracciones
                infraccion_detectada = False
                
                # 1. Exceso de velocidad
                exceso, velocidad = self.detectar_exceso_velocidad(vehiculo_id, self.frame_count)
                if exceso and self.puede_registrar_infraccion(vehiculo_id, 'EXCESO_VEL'):
                    self.registrar_infraccion_async(
                        'EXCESO_VEL', frame, placa_vehiculo,
                        velocidad=velocidad, confianza=conf, imagen_placa=roi_placa
                    )
                    self.dibujar_infraccion(frame_display, x1, y1, x2, y2, 
                                          f"EXCESO: {velocidad:.0f} km/h", placa_vehiculo, (0, 0, 255))
                    infraccion_detectada = True
                
                # 2. Luz roja
                if luz_roja and self.puede_registrar_infraccion(vehiculo_id, 'LUZ_ROJA'):
                    self.registrar_infraccion_async(
                        'LUZ_ROJA', frame, placa_vehiculo,
                        confianza=conf, imagen_placa=roi_placa
                    )
                    self.dibujar_infraccion(frame_display, x1, y1, x2, y2,
                                          "LUZ ROJA", placa_vehiculo, (0, 0, 255))
                    infraccion_detectada = True
                
                # 3. Invasión de carril
                invasion = self.detectar_invasion_carril(frame, x1, y1, x2, y2)
                if invasion and self.puede_registrar_infraccion(vehiculo_id, 'INVASION_CARRIL'):
                    self.registrar_infraccion_async(
                        'INVASION_CARRIL', frame, placa_vehiculo,
                        confianza=conf, imagen_placa=roi_placa
                    )
                    self.dibujar_infraccion(frame_display, x1, y1, x2, y2,
                                          "INVASION CARRIL", placa_vehiculo, (0, 165, 255))
                    infraccion_detectada = True
                
                if not infraccion_detectada:
                    # Dibujar detección normal
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_display, f"{cls} {conf:.2f}",
                              (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame_display, f"Placa: {placa_vehiculo}",
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Actualizar tracking
                if vehiculo_id not in self.vehiculos_trackeados:
                    self.vehiculos_trackeados[vehiculo_id] = {
                        'frame': self.frame_count,
                        'placa': placa_vehiculo
                    }
        
        # Dibujar semáforo si está en rojo
        if luz_roja and coords_semaforo:
            x1, y1, x2, y2 = [int(c / escala) for c in coords_semaforo]
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame_display, "SEMAFORO ROJO", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        self.dibujar_info_sistema(frame_display, fps_actual)
        
        return frame_display
    
    def dibujar_infraccion(self, frame, x1, y1, x2, y2, texto, placa, color):
        """Dibuja una infracción detectada"""
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, texto, (x1, y1-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Placa: {placa}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def dibujar_info_sistema(self, frame, fps):
        """Dibuja información del sistema"""
        cv2.rectangle(frame, (5, 5), (450, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (450, 120), (0, 255, 0), 2)
        
        cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {self.frame_count}",
                   (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehiculos: {len(self.vehiculos_trackeados)}",
                   (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Placas Peruanas: {len(self.placas_detectadas)}",
                   (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Formato: A1B-234",
                   (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def iniciar_deteccion(self):
        """Inicia detección en tiempo real"""
        print("\n🎥 Iniciando detección optimizada...")
        print("📋 Infracciones: Luz Roja | Exceso Velocidad | Invasión Carril")
        print("🇵🇪 Formato de placa: A1B-234 (Perú)")
        print("Presiona 'q' para salir\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Error al capturar frame")
                    break
                
                frame_procesado = self.procesar_frame(frame)
                
                cv2.imshow('Detector Placas Perú - Tesis (Optimizado)', frame_procesado)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️  Detección interrumpida")
        
        finally:
            self.detener()
    
    def detener(self):
        """Libera recursos"""
        print("\n🛑 Deteniendo sistema...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Sistema detenido")


def main():
    """Función principal"""
    print("=" * 80)
    print("🚦 DETECTOR OPTIMIZADO DE PLACAS PERUANAS")
    print("🇵🇪 Formato: A1B-234 (3 alfanuméricos + guión + 3 números)")
    print("📚 Proyecto de Tesis - IoT + Visión AI + ML")
    print("=" * 80)
    print()
    
    try:
        detector = DetectorPlacasPeru(camara_id=0, skip_frames=2, usar_gpu=True)
        detector.iniciar_deteccion()
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

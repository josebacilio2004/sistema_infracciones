"""
Sistema de detección mejorado con reconocimiento de placas peruanas
Optimizado para máximo rendimiento en FPS
Integra YOLOv8 + EasyOCR + Detección de 3 infracciones principales
"""
import os
import sys
import django
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import re
import threading
from collections import deque
import time

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

class DetectorWebcamMejorado:
    """Detector optimizado de infracciones con reconocimiento de placas peruanas"""
    
    def __init__(self, fuente_video=0, skip_frames=2, usar_gpu=True):
        print("🚀 Inicializando sistema de detección mejorado...")
        
        self.skip_frames = skip_frames
        self.frame_count = 0
        
        # Cargar modelo YOLO optimizado
        print("📦 Cargando YOLOv8n...")
        self.modelo_yolo = YOLO('yolov8n.pt')
        self.modelo_yolo.fuse()  # Fusionar capas para mejor rendimiento
        print("✅ Modelo YOLO cargado y optimizado")
        
        # Inicializar OCR para placas peruanas
        print("📝 Cargando EasyOCR para placas peruanas...")
        self.reader = easyocr.Reader(['en'], gpu=usar_gpu)
        self.ocr_activo = False
        self.ocr_thread = None
        self.ocr_queue = deque(maxlen=5)
        print("✅ OCR inicializado")
        
        # Inicializar predictor ML
        try:
            self.predictor_ml = PredictorRiesgo()
            print("✅ Predictor ML inicializado")
        except Exception as e:
            print(f"⚠️  Predictor ML no disponible: {e}")
            self.predictor_ml = None
        
        # Configurar fuente de video
        self.fuente_video = fuente_video
        self.cap = cv2.VideoCapture(fuente_video)
        
        if not self.cap.isOpened():
            raise Exception(f"❌ No se pudo abrir la fuente de video: {fuente_video}")
        
        # Configurar resolución óptima
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        ancho = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Fuente de video conectada ({ancho}x{alto})")
        
        # Obtener o crear cámara en BD
        self.camara_db, created = Camara.objects.get_or_create(
            ubicacion="Webcam Local - Pruebas Mejoradas",
            defaults={
                'ip': '127.0.0.1',
                'descripcion': 'Cámara de prueba con OCR y detección optimizada',
                'activa': True,
                'tipo_fuente': 'WEBCAM'
            }
        )
        if created:
            print("✅ Cámara registrada en base de datos")
        
        # Configuración de detección
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.vehiculos_trackeados = {}
        self.placas_detectadas = {}
        self.ultimas_infracciones = deque(maxlen=100)
        
        # Límites y configuración
        self.LIMITE_VELOCIDAD = 60  # km/h
        self.DISTANCIA_METROS = 20
        self.COOLDOWN_INFRACCION = 5  # segundos entre infracciones del mismo vehículo
        self.ultimo_registro = {}
        
        # Métricas de rendimiento
        self.fps_real = deque(maxlen=30)
        self.tiempo_inicio = time.time()
        
        # Crear carpetas para evidencias
        self.carpeta_evidencias = BASE_DIR / 'media' / 'infracciones' / 'imagenes'
        self.carpeta_placas = BASE_DIR / 'media' / 'infracciones' / 'placas'
        self.carpeta_evidencias.mkdir(parents=True, exist_ok=True)
        self.carpeta_placas.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Sistema listo - Skip frames: {skip_frames}, GPU: {usar_gpu}")
        print("🎯 Infracciones monitoreadas: Luz Roja, Exceso Velocidad, Invasión Carril\n")
    
    def limpiar_placa_peruana(self, texto):
        """Limpia y valida el texto de la placa peruana (formato A1B-234)"""
        # Remover espacios y caracteres especiales
        texto = re.sub(r'[^A-Z0-9]', '', texto.upper())
        
        # Validar formato peruano: 3 caracteres alfanuméricos + 3 números
        # Ejemplos: A1B234, ABC123, A12345
        if len(texto) == 6:
            # Formato: ABC123 o A1B234
            if re.match(r'^[A-Z0-9]{3}[0-9]{3}$', texto):
                return f"{texto[:3]}-{texto[3:]}"
        elif len(texto) == 7:
            # Ya tiene guión
            if re.match(r'^[A-Z0-9]{3}-?[0-9]{3}$', texto):
                return texto if '-' in texto else f"{texto[:3]}-{texto[3:]}"
        
        return None
    
    def detectar_placa_peruana(self, frame, x1, y1, x2, y2):
        """Detecta y lee placas peruanas usando OCR optimizado"""
        try:
            # Expandir región de interés
            h, w = frame.shape[:2]
            margen = 30
            y1_exp = max(0, y1 - margen)
            y2_exp = min(h, y2 + margen)
            x1_exp = max(0, x1 - margen)
            x2_exp = min(w, x2 + margen)
            
            roi = frame[y1_exp:y2_exp, x1_exp:x2_exp]
            
            if roi.size == 0:
                return None, None
            
            # Preprocesar para placas blancas peruanas
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Mejorar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Threshold adaptativo para placas blancas
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Leer texto con OCR
            resultados = self.reader.readtext(thresh, detail=1, paragraph=False)
            
            for (bbox, texto, confianza) in resultados:
                if confianza > 0.4:  # Umbral más bajo para placas
                    placa_limpia = self.limpiar_placa_peruana(texto)
                    if placa_limpia:
                        return placa_limpia, confianza
            
            return None, None
            
        except Exception as e:
            return None, None
    
    def detectar_luz_roja(self, frame, resultados):
        """Detecta si hay un semáforo en rojo"""
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
                total_pixeles = semaforo.shape[0] * semaforo.shape[1]
                
                if pixeles_rojos > total_pixeles * 0.1:  # 10% de pixeles rojos
                    return True, (x1, y1, x2, y2)
        
        return False, None
    
    def detectar_exceso_velocidad(self, vehiculo_id, frame_actual):
        """Detecta exceso de velocidad basado en tracking"""
        if vehiculo_id in self.vehiculos_trackeados:
            datos = self.vehiculos_trackeados[vehiculo_id]
            frame_anterior = datos['frame']
            frame_diff = frame_actual - frame_anterior
            
            tiempo_segundos = frame_diff / self.fps
            
            if tiempo_segundos > 0.5:  # Mínimo medio segundo
                velocidad = (self.DISTANCIA_METROS / tiempo_segundos) * 3.6
                
                # Filtrar valores irreales
                if self.LIMITE_VELOCIDAD < velocidad < 150:
                    return True, velocidad
        
        return False, 0
    
    def detectar_invasion_carril(self, frame, x1, y1, x2, y2):
        """Detecta invasión de carril (simplificado)"""
        centro_x = (x1 + x2) // 2
        centro_y = (y1 + y2) // 2
        
        h, w = frame.shape[:2]
        linea_central = w // 2
        margen_carril = w // 8
        
        # Si el vehículo está muy cerca de la línea central
        if abs(centro_x - linea_central) < margen_carril:
            return True
        
        return False
    
    def puede_registrar_infraccion(self, vehiculo_id, tipo_codigo):
        """Verifica si puede registrar una infracción (cooldown)"""
        clave = f"{vehiculo_id}_{tipo_codigo}"
        
        if clave in self.ultimo_registro:
            tiempo_transcurrido = (datetime.now() - self.ultimo_registro[clave]).total_seconds()
            if tiempo_transcurrido < self.COOLDOWN_INFRACCION:
                return False
        
        self.ultimo_registro[clave] = datetime.now()
        return True
    
    def registrar_infraccion(self, tipo_codigo, frame, vehiculo_placa="DESCONOCIDA",
                            velocidad=None, confianza=0.85, imagen_placa=None):
        """Registra una infracción en la base de datos"""
        try:
            # Obtener o crear vehículo
            vehiculo, _ = Vehiculo.objects.get_or_create(
                placa=vehiculo_placa,
                defaults={'tipo_vehiculo': 'AUTO'}
            )
            
            # Obtener tipo de infracción
            tipo_infraccion = TipoInfraccion.objects.filter(codigo=tipo_codigo).first()
            if not tipo_infraccion:
                print(f"⚠️  Tipo de infracción {tipo_codigo} no encontrado")
                return None
            
            # Guardar imagen de evidencia
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"{tipo_codigo}_{vehiculo_placa}_{timestamp}.jpg"
            ruta_imagen = self.carpeta_evidencias / nombre_archivo
            cv2.imwrite(str(ruta_imagen), frame)
            
            # Guardar imagen de placa si existe
            ruta_placa_rel = None
            if imagen_placa is not None:
                nombre_placa = f"placa_{vehiculo_placa}_{timestamp}.jpg"
                ruta_placa = self.carpeta_placas / nombre_placa
                cv2.imwrite(str(ruta_placa), imagen_placa)
                ruta_placa_rel = f'infracciones/placas/{nombre_placa}'
            
            # Crear infracción
            infraccion = Infraccion.objects.create(
                vehiculo=vehiculo,
                tipo_infraccion=tipo_infraccion,
                camara=self.camara_db,
                ubicacion=self.camara_db.ubicacion,
                velocidad_detectada=int(velocidad) if velocidad else None,
                velocidad_maxima=self.LIMITE_VELOCIDAD if velocidad else None,
                imagen_principal=f'infracciones/imagenes/{nombre_archivo}',
                imagen_placa=ruta_placa_rel,
                confianza_deteccion=confianza * 100,
                modelo_ia_version='YOLOv8n + EasyOCR',
                estado='DETECTADA'
            )
            
            # Registrar evento
            EventoDeteccion.objects.create(
                camara=self.camara_db,
                tipo_evento='INFRACCION_DETECTADA',
                datos_evento={
                    'tipo': tipo_codigo,
                    'placa': vehiculo_placa,
                    'velocidad': velocidad,
                    'confianza': confianza,
                    'ocr_usado': imagen_placa is not None
                }
            )
            
            # Actualizar predicción ML
            if self.predictor_ml:
                try:
                    prediccion = self.predictor_ml.predecir_riesgo_vehiculo(vehiculo_placa)
                    print(f"📊 ML: Riesgo {prediccion['nivel_riesgo']} "
                          f"({prediccion['probabilidad_reincidencia']:.1f}%)")
                except:
                    pass
            
            self.ultimas_infracciones.append({
                'tipo': tipo_codigo,
                'placa': vehiculo_placa,
                'timestamp': datetime.now()
            })
            
            print(f"✅ Infracción registrada: {tipo_infraccion.nombre} - {vehiculo_placa}")
            return infraccion
            
        except Exception as e:
            print(f"❌ Error al registrar infracción: {e}")
            return None
    
    def procesar_frame(self, frame):
        """Procesa un frame y detecta infracciones"""
        tiempo_frame_inicio = time.time()
        self.frame_count += 1
        
        # Skip frames para mejor rendimiento
        if self.frame_count % (self.skip_frames + 1) != 0:
            return frame
        
        frame_display = frame.copy()
        
        # Ejecutar detección YOLO con tracking
        resultados = self.modelo_yolo.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.5,
            iou=0.5
        )
        
        if not resultados or len(resultados[0].boxes) == 0:
            return frame_display
        
        # Detectar luz roja
        luz_roja, coords_semaforo = self.detectar_luz_roja(frame, resultados)
        
        # Procesar cada vehículo detectado
        for box in resultados[0].boxes:
            cls = self.modelo_yolo.names[int(box.cls)]
            conf = float(box.conf[0])
            
            # Solo procesar vehículos
            if cls not in ['car', 'truck', 'bus', 'motorcycle']:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehiculo_id = int(box.id[0]) if box.id is not None else None
            
            if not vehiculo_id:
                continue
            
            # Detectar placa cada 30 frames
            placa_detectada = None
            if self.frame_count % 30 == 0 or vehiculo_id not in self.placas_detectadas:
                placa_detectada, conf_placa = self.detectar_placa_peruana(frame, x1, y1, x2, y2)
                if placa_detectada:
                    self.placas_detectadas[vehiculo_id] = placa_detectada
                    print(f"🚗 Placa peruana: {placa_detectada} ({conf_placa:.2f})")
            
            placa_vehiculo = self.placas_detectadas.get(vehiculo_id, f"VEH-{vehiculo_id:04d}")
            
            # Detectar exceso de velocidad
            exceso, velocidad = self.detectar_exceso_velocidad(vehiculo_id, self.frame_count)
            
            if exceso and self.puede_registrar_infraccion(vehiculo_id, 'EXCESO_VEL'):
                roi_placa = frame[y1:y2, x1:x2] if placa_detectada else None
                self.registrar_infraccion(
                    'EXCESO_VEL',
                    frame,
                    placa_vehiculo,
                    velocidad=velocidad,
                    confianza=conf,
                    imagen_placa=roi_placa
                )
                
                # Dibujar alerta
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame_display, f"EXCESO: {velocidad:.0f} km/h",
                          (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame_display, f"{placa_vehiculo}",
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Resetear tracking
                if vehiculo_id in self.vehiculos_trackeados:
                    del self.vehiculos_trackeados[vehiculo_id]
            else:
                # Actualizar tracking
                if vehiculo_id not in self.vehiculos_trackeados:
                    self.vehiculos_trackeados[vehiculo_id] = {
                        'frame': self.frame_count,
                        'placa': placa_vehiculo
                    }
                
                # Detectar invasión de carril
                invasion = self.detectar_invasion_carril(frame, x1, y1, x2, y2)
                
                if invasion and self.puede_registrar_infraccion(vehiculo_id, 'INVASION_CARRIL'):
                    roi_placa = frame[y1:y2, x1:x2] if placa_detectada else None
                    self.registrar_infraccion(
                        'INVASION_CARRIL',
                        frame,
                        placa_vehiculo,
                        confianza=conf,
                        imagen_placa=roi_placa
                    )
                    
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 165, 255), 3)
                    cv2.putText(frame_display, "INVASION CARRIL",
                              (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(frame_display, f"{placa_vehiculo}",
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    # Dibujar detección normal
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_display, f"{cls} {conf:.2f}",
                              (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame_display, f"{placa_vehiculo}",
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Detectar luz roja
            if luz_roja and self.puede_registrar_infraccion(vehiculo_id, 'LUZ_ROJA'):
                roi_placa = frame[y1:y2, x1:x2] if placa_detectada else None
                self.registrar_infraccion(
                    'LUZ_ROJA',
                    frame,
                    placa_vehiculo,
                    confianza=conf,
                    imagen_placa=roi_placa
                )
                
                cv2.putText(frame_display, "LUZ ROJA!",
                          (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calcular FPS real
        tiempo_frame = time.time() - tiempo_frame_inicio
        fps_actual = 1.0 / tiempo_frame if tiempo_frame > 0 else 0
        self.fps_real.append(fps_actual)
        fps_promedio = sum(self.fps_real) / len(self.fps_real)
        
        # Dibujar información del sistema
        cv2.rectangle(frame_display, (5, 5), (450, 110), (0, 0, 0), -1)
        cv2.rectangle(frame_display, (5, 5), (450, 110), (0, 255, 0), 2)
        
        cv2.putText(frame_display, f"Frame: {self.frame_count} | FPS: {fps_promedio:.1f}",
                   (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_display, f"Vehiculos: {len(self.vehiculos_trackeados)}",
                   (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_display, f"Placas Peruanas: {len(self.placas_detectadas)}",
                   (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_display, f"Infracciones: {len(self.ultimas_infracciones)}",
                   (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if luz_roja and coords_semaforo:
            x1, y1, x2, y2 = coords_semaforo
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame_display, "SEMAFORO ROJO", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame_display
    
    def iniciar_deteccion(self):
        """Inicia el loop de detección en tiempo real"""
        print("\n🎥 Iniciando detección en tiempo real...")
        print("Presiona 'q' para salir\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Error al capturar frame")
                    break
                
                # Procesar frame
                frame_procesado = self.procesar_frame(frame)
                
                # Mostrar resultado
                cv2.imshow('Sistema de Detección - Tesis (Optimizado)', frame_procesado)
                
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️  Detección interrumpida por usuario")
        
        finally:
            self.detener()
    
    def detener(self):
        """Libera recursos"""
        print("\n🛑 Deteniendo sistema...")
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Estadísticas finales
        tiempo_total = time.time() - self.tiempo_inicio
        fps_promedio = sum(self.fps_real) / len(self.fps_real) if self.fps_real else 0
        
        print(f"\n📊 Estadísticas de la sesión:")
        print(f"   - Tiempo total: {tiempo_total:.1f}s")
        print(f"   - Frames procesados: {self.frame_count}")
        print(f"   - FPS promedio: {fps_promedio:.1f}")
        print(f"   - Vehículos detectados: {len(self.vehiculos_trackeados)}")
        print(f"   - Placas peruanas: {len(self.placas_detectadas)}")
        print(f"   - Infracciones registradas: {len(self.ultimas_infracciones)}")
        print("✅ Sistema detenido correctamente")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Detección de Infracciones Mejorado')
    parser.add_argument('--fuente', type=str, default='0',
                       help='Fuente de video: 0 (webcam), URL (IP/Iriun), o ruta de archivo')
    parser.add_argument('--skip-frames', type=int, default=2,
                       help='Frames a saltar (0=todos, 2=1 de cada 3, 4=1 de cada 5)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Desactivar GPU (usar CPU)')
    
    args = parser.parse_args()
    
    # Convertir fuente
    fuente = args.fuente
    if fuente.isdigit():
        fuente = int(fuente)
    
    print("=" * 80)
    print("🚦 SISTEMA DE DETECCIÓN DE INFRACCIONES CON IA MEJORADO")
    print("📚 Proyecto de Tesis - IoT + Visión AI + ML + OCR")
    print("🇵🇪 Optimizado para placas peruanas (formato A1B-234)")
    print("=" * 80)
    print(f"\n📹 Fuente: {fuente}")
    print(f"⚡ Skip frames: {args.skip_frames}")
    print(f"🖥️  GPU: {'No' if args.no_gpu else 'Sí'}\n")
    
    try:
        detector = DetectorWebcamMejorado(
            fuente_video=fuente,
            skip_frames=args.skip_frames,
            usar_gpu=not args.no_gpu
        )
        detector.iniciar_deteccion()
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

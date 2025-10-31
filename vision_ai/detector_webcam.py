"""
Sistema de detección en tiempo real con webcam
Integra Visión Artificial con Django para pruebas rápidas
"""
import os
import sys
import django
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# Configurar Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

from ultralytics import YOLO
from infracciones.models import Infraccion, Vehiculo, TipoInfraccion, EventoDeteccion
from camaras.models import Camara
from ml_predicciones.predictor import PredictorRiesgo

class DetectorWebcam:
    """Detector de infracciones en tiempo real usando webcam"""
    
    def __init__(self, camara_id=0):
        print("🚀 Inicializando sistema de detección...")
        
        # Cargar modelo YOLO
        self.modelo_yolo = YOLO('yolov8n.pt')
        print("✅ Modelo YOLO cargado")
        
        # Inicializar predictor ML
        self.predictor_ml = PredictorRiesgo()
        print("✅ Predictor ML inicializado")
        
        # Configurar cámara
        self.cap = cv2.VideoCapture(camara_id)
        if not self.cap.isOpened():
            raise Exception("❌ No se pudo abrir la webcam")
        print("✅ Webcam conectada")
        
        # Obtener o crear cámara en BD
        self.camara_db, created = Camara.objects.get_or_create(
            ubicacion="Webcam Local - Pruebas",
            defaults={'ip': '127.0.0.1', 'descripcion': 'Cámara de prueba local'}
        )
        if created:
            print("✅ Cámara registrada en base de datos")
        
        # Configuración de detección
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.frame_count = 0
        self.detecciones_vehiculos = {}
        
        # Límites de velocidad
        self.LIMITE_VELOCIDAD = 60  # km/h
        self.DISTANCIA_METROS = 20
        
        # Crear carpetas para evidencias
        self.carpeta_evidencias = BASE_DIR / 'media' / 'infracciones' / 'imagenes'
        self.carpeta_evidencias.mkdir(parents=True, exist_ok=True)
        
        print("✅ Sistema listo para detectar infracciones\n")
    
    def detectar_luz_roja(self, frame, resultados):
        """Detecta si un vehículo cruza con luz roja"""
        for box in resultados[0].boxes:
            cls = self.modelo_yolo.names[int(box.cls)]
            
            if cls == 'traffic light':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                semaforo = frame[y1:y2, x1:x2]
                
                if semaforo.size == 0:
                    continue
                
                # Detectar color rojo en HSV
                hsv = cv2.cvtColor(semaforo, cv2.COLOR_BGR2HSV)
                
                # Rango para rojo (dos rangos porque el rojo está en los extremos)
                rojo_bajo1 = np.array([0, 100, 100])
                rojo_alto1 = np.array([10, 255, 255])
                rojo_bajo2 = np.array([160, 100, 100])
                rojo_alto2 = np.array([180, 255, 255])
                
                mascara1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
                mascara2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
                mascara_roja = cv2.bitwise_or(mascara1, mascara2)
                
                pixeles_rojos = cv2.countNonZero(mascara_roja)
                
                if pixeles_rojos > 50:
                    return True, (x1, y1, x2, y2)
        
        return False, None
    
    def detectar_exceso_velocidad(self, vehiculo_id, frame_actual):
        """Detecta exceso de velocidad basado en tracking"""
        if vehiculo_id in self.detecciones_vehiculos:
            frame_anterior = self.detecciones_vehiculos[vehiculo_id]
            frame_diff = frame_actual - frame_anterior
            
            tiempo_segundos = frame_diff / self.fps
            
            if tiempo_segundos > 0:
                velocidad = (self.DISTANCIA_METROS / tiempo_segundos) * 3.6
                
                if velocidad > self.LIMITE_VELOCIDAD and velocidad < 200:  # Filtrar valores irreales
                    return True, velocidad
        
        return False, 0
    
    def registrar_infraccion(self, tipo_codigo, frame, vehiculo_placa="DESCONOCIDA", 
                            velocidad=None, confianza=0.85):
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
                modelo_ia_version='YOLOv8n',
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
                    'confianza': confianza
                }
            )
            
            # Actualizar predicción de riesgo con ML
            try:
                prediccion = self.predictor_ml.predecir_riesgo_vehiculo(vehiculo_placa)
                print(f"📊 Predicción ML: Riesgo {prediccion['nivel_riesgo']} "
                      f"({prediccion['probabilidad_reincidencia']:.1f}% reincidencia)")
            except Exception as e:
                print(f"⚠️  Error en predicción ML: {e}")
            
            print(f"✅ Infracción registrada: {tipo_infraccion.nombre} - {vehiculo_placa}")
            return infraccion
            
        except Exception as e:
            print(f"❌ Error al registrar infracción: {e}")
            return None
    
    def procesar_frame(self, frame):
        """Procesa un frame y detecta infracciones"""
        self.frame_count += 1
        
        # Ejecutar detección YOLO con tracking
        resultados = self.modelo_yolo.track(frame, persist=True, verbose=False)
        
        if not resultados or len(resultados[0].boxes) == 0:
            return frame
        
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
            
            # Obtener ID de tracking
            vehiculo_id = int(box.id[0]) if box.id is not None else None
            
            if vehiculo_id:
                # Detectar exceso de velocidad
                exceso, velocidad = self.detectar_exceso_velocidad(vehiculo_id, self.frame_count)
                
                if exceso:
                    placa = f"VEH-{vehiculo_id:04d}"
                    self.registrar_infraccion(
                        'EXCESO_VEL',
                        frame,
                        placa,
                        velocidad=velocidad,
                        confianza=conf
                    )
                    
                    # Dibujar alerta en frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"EXCESO: {velocidad:.0f} km/h", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (0, 0, 255), 2)
                    
                    # Resetear tracking para este vehículo
                    del self.detecciones_vehiculos[vehiculo_id]
                else:
                    # Actualizar tracking
                    if vehiculo_id not in self.detecciones_vehiculos:
                        self.detecciones_vehiculos[vehiculo_id] = self.frame_count
                    
                    # Dibujar detección normal
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls} {conf:.2f}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                
                # Detectar luz roja
                if luz_roja:
                    placa = f"VEH-{vehiculo_id:04d}"
                    self.registrar_infraccion(
                        'LUZ_ROJA',
                        frame,
                        placa,
                        confianza=conf
                    )
                    
                    cv2.putText(frame, "LUZ ROJA!", 
                              (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (0, 0, 255), 2)
        
        # Dibujar información del sistema
        cv2.putText(frame, f"Frame: {self.frame_count} | FPS: {self.fps}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehiculos: {len(self.detecciones_vehiculos)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if luz_roja and coords_semaforo:
            x1, y1, x2, y2 = coords_semaforo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "SEMAFORO ROJO", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
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
                cv2.imshow('Sistema de Detección de Infracciones - Tesis', frame_procesado)
                
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
        print("✅ Sistema detenido correctamente")


def main():
    """Función principal"""
    print("=" * 60)
    print("🚦 SISTEMA DE DETECCIÓN DE INFRACCIONES CON IA")
    print("📚 Proyecto de Tesis - Integración IoT + Visión AI + ML")
    print("=" * 60)
    print()
    
    try:
        detector = DetectorWebcam(camara_id=0)
        detector.iniciar_deteccion()
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

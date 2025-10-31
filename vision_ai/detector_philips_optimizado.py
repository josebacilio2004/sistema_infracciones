"""
Sistema Optimizado para Cámara Philips 1080P - Detección de Infracciones Perú
"""
import os
import sys
import django
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Configurar Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

from ultralytics import YOLO
import pytesseract
from infracciones.models import Infraccion, Vehiculo, TipoInfraccion, EventoDeteccion
from camaras.models import Camara

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectorPhilipsOptimizado:
    """Sistema optimizado para cámara Philips 1080P"""
    
    def __init__(self, camara_id=0):
        print("🚀 INICIANDO SISTEMA OPTIMIZADO PHILIPS 1080P")
        print("=" * 60)
        
        # Configuración específica para Philips 1080P
        self.resolucion = (1920, 1080)  # Full HD
        self.fps_objetivo = 30
        
        # Cargar modelos
        self._cargar_modelos()
        
        # Configurar cámara Philips
        self._configurar_camara_philips(camara_id)
        
        # Configurar base de datos
        self._configurar_bd()
        
        # Variables de tracking
        self.frame_count = 0
        self.vehiculos_trackeados = {}
        self.infracciones_registradas = []
        
        # Estadísticas en tiempo real
        self.estadisticas = {
            'vehiculos_detectados': 0,
            'placas_reconocidas': 0,
            'infracciones_registradas': 0,
            'fps_real': 0
        }
        
        # Tiempo para cálculo de FPS
        self.tiempo_inicio = datetime.now()
        
        print("✅ SISTEMA OPTIMIZADO INICIALIZADO")
        print(f"   • Cámara: Philips 1080P")
        print(f"   • Resolución: {self.resolucion[0]}x{self.resolucion[1]}")
        print(f"   • FPS objetivo: {self.fps_objetivo}")
        print("=" * 60)
    
    def _cargar_modelos(self):
        """Carga modelos optimizados para 1080P"""
        try:
            print("📦 Cargando YOLOv8n (optimizado)...")
            self.modelo_yolo = YOLO('yolov8n.pt')
            
            # Optimizar modelo para velocidad
            self.modelo_yolo.overrides['conf'] = 0.5  # Confianza mínima
            self.modelo_yolo.overrides['iou'] = 0.45  # IoU threshold
            self.modelo_yolo.overrides['agnostic_nms'] = False
            self.modelo_yolo.overrides['max_det'] = 20  # Máximo de detecciones
            
            # Configurar Tesseract
            pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Bacilio\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
            
            print("✅ Modelos cargados y optimizados")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            raise
    
    def _configurar_camara_philips(self, camara_id):
        """Configura la cámara Philips 1080P"""
        try:
            self.cap = cv2.VideoCapture(camara_id, cv2.CAP_DSHOW)  # Usar DirectShow para mejor compatibilidad
            
            if not self.cap.isOpened():
                raise Exception("❌ No se pudo conectar a la cámara Philips")
            
            # Configurar propiedades de la cámara Philips
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolucion[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolucion[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_objetivo)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Desactivar autofocus
            self.cap.set(cv2.CAP_PROP_FOCUS, 50)     # Focus fijo
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 60) # Brillo medio
            self.cap.set(cv2.CAP_PROP_CONTRAST, 55)   # Contraste medio
            
            # Verificar configuración
            ancho_real = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto_real = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_real = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"📷 Cámara Philips configurada: {ancho_real}x{alto_real} @ {fps_real}FPS")
            
            # Ajustar resolución si es necesario
            if ancho_real != self.resolucion[0]:
                self.resolucion = (ancho_real, alto_real)
                print(f"🔧 Resolución ajustada a: {self.resolucion}")
            
        except Exception as e:
            logger.error(f"❌ Error configurando cámara Philips: {e}")
            raise
    
    def _configurar_bd(self):
        """Configura la base de datos"""
        try:
            self.camara_db, created = Camara.objects.get_or_create(
                ubicacion="Puesto de Control - Philips 1080P",
                defaults={
                    'ip': 'localhost',
                    'descripcion': 'Cámara Philips SPL6506BM Full HD 1080P',
                    'resolucion': '1080P',
                    'activa': True,
                    'fps': 30
                }
            )
            
            # Crear carpeta para evidencias
            self.carpeta_evidencias = BASE_DIR / 'media' / 'infracciones'
            self.carpeta_evidencias.mkdir(parents=True, exist_ok=True)
            
            print("✅ Base de datos configurada")
            
        except Exception as e:
            logger.error(f"❌ Error configurando BD: {e}")
            raise
    
    def detectar_placas_philips(self, frame_vehiculo):
        """Detección optimizada de placas para cámara Philips"""
        try:
            # Redimensionar para mejor procesamiento
            altura, ancho = frame_vehiculo.shape[:2]
            if ancho > 400:
                scale = 400 / ancho
                nuevo_ancho = 400
                nueva_altura = int(altura * scale)
                frame_vehiculo = cv2.resize(frame_vehiculo, (nuevo_ancho, nueva_altura))
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame_vehiculo, cv2.COLOR_BGR2GRAY)
            
            # Mejorar contraste para placas
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_enhanced = clahe.apply(gray)
            
            # Umbral adaptativo
            thresh = cv2.adaptiveThreshold(
                gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Operaciones morfológicas para mejorar texto
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Configuración optimizada para placas peruanas
            configs = [
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 13'
            ]
            
            for config in configs:
                try:
                    texto = pytesseract.image_to_string(morph, config=config)
                    texto_limpio = self._limpiar_placa_peruana(texto)
                    
                    if texto_limpio and self._validar_formato_placa(texto_limpio):
                        logger.info(f"🔍 Placa detectada: {texto_limpio}")
                        self.estadisticas['placas_reconocidas'] += 1
                        return texto_limpio
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error detectando placa: {e}")
            return None
    
    def _limpiar_placa_peruana(self, texto):
        """Limpia y formatea placa peruana"""
        # Remover caracteres no alfanuméricos
        texto_limpio = ''.join(c for c in texto.upper() if c.isalnum())
        
        # Formatear placas comunes en Perú
        if len(texto_limpio) == 6:  # AAA111 -> AAA-111
            return f"{texto_limpio[:3]}-{texto_limpio[3:]}"
        elif len(texto_limpio) == 7 and texto_limpio[3].isalpha():  # AAA1111 -> AAA-1111
            return f"{texto_limpio[:3]}-{texto_limpio[3:]}"
        
        return texto_limpio if len(texto_limpio) >= 4 else None
    
    def _validar_formato_placa(self, placa):
        """Valida formato de placa peruana"""
        placa_sin_guion = placa.replace('-', '')
        
        # Debe tener letras y números
        tiene_letras = any(c.isalpha() for c in placa_sin_guion)
        tiene_numeros = any(c.isdigit() for c in placa_sin_guion)
        
        if not (tiene_letras and tiene_numeros):
            return False
        
        # Longitudes típicas de placas peruanas
        if len(placa_sin_guion) < 5 or len(placa_sin_guion) > 7:
            return False
        
        return True
    
    def detectar_vehiculos_y_placas(self, frame):
        """Detección principal de vehículos y placas"""
        infracciones = []
        
        try:
            # Ejecutar YOLO con tracking
            resultados = self.modelo_yolo.track(frame, persist=True, verbose=False)
            
            if not resultados or len(resultados[0].boxes) == 0:
                return infracciones
            
            # Procesar cada detección
            for box in resultados[0].boxes:
                cls = self.modelo_yolo.names[int(box.cls)]
                conf = float(box.conf[0])
                
                # Filtrar solo vehículos
                if cls not in ['car', 'truck', 'bus', 'motorcycle']:
                    continue
                
                self.estadisticas['vehiculos_detectados'] += 1
                
                # Obtener coordenadas
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Extraer región del vehículo
                roi_vehiculo = frame[y1:y2, x1:x2]
                
                if roi_vehiculo.size == 0:
                    continue
                
                # Intentar detectar placa
                placa = self.detectar_placas_philips(roi_vehiculo)
                
                # Detectar infracciones básicas
                infraccion = self._detectar_infracciones_basicas(box, placa, conf)
                if infraccion:
                    infracciones.append(infraccion)
            
            return infracciones
            
        except Exception as e:
            logger.error(f"❌ Error en detección principal: {e}")
            return []
    
    def _detectar_infracciones_basicas(self, box, placa, confianza):
        """Detecta infracciones básicas para pruebas"""
        infraccion = None
        
        # Simular detección de exceso de velocidad (para pruebas)
        # En un sistema real, esto usaría tracking entre frames
        if box.id is not None:
            vehiculo_id = int(box.id[0])
            
            # Simular velocidad aleatoria para pruebas (30-80 km/h)
            import random
            velocidad = random.randint(30, 80)
            
            if velocidad > 60:  # Límite de 60 km/h
                infraccion = {
                    'tipo': 'EXCESO_VELOCIDAD',
                    'vehiculo_id': vehiculo_id,
                    'placa': placa or f"VEH-{vehiculo_id:04d}",
                    'velocidad': velocidad,
                    'confianza': confianza,
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'timestamp': datetime.now()
                }
        
        return infraccion
    
    def registrar_infraccion_db(self, infraccion, frame):
        """Registra infracción en la base de datos"""
        try:
            # Obtener tipo de infracción
            tipo_infraccion, created = TipoInfraccion.objects.get_or_create(
                codigo='A01',  # Exceso de velocidad
                defaults={
                    'nombre': 'Exceso de velocidad',
                    'gravedad': 'MEDIA',
                    'monto_multa': 360.00
                }
            )
            
            # Obtener o crear vehículo
            placa = infraccion.get('placa', 'DESCONOCIDA')
            vehiculo, _ = Vehiculo.objects.get_or_create(
                placa=placa,
                defaults={'tipo_vehiculo': 'AUTO'}
            )
            
            # Guardar imagen de evidencia
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            nombre_archivo = f"A01_{placa}_{timestamp}.jpg"
            ruta_imagen = self.carpeta_evidencias / nombre_archivo
            
            # Recortar región del vehículo
            if 'bbox' in infraccion:
                x1, y1, x2, y2 = map(int, infraccion['bbox'])
                # Asegurar coordenadas dentro del frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    evidencia_frame = frame[y1:y2, x1:x2]
                    cv2.imwrite(str(ruta_imagen), evidencia_frame)
            
            # Crear registro de infracción
            infraccion_db = Infraccion.objects.create(
                vehiculo=vehiculo,
                tipo_infraccion=tipo_infraccion,
                camara=self.camara_db,
                ubicacion=self.camara_db.ubicacion,
                velocidad_detectada=infraccion.get('velocidad'),
                velocidad_maxima=60,
                imagen_principal=f'infracciones/{nombre_archivo}',
                confianza_deteccion=infraccion.get('confianza', 0.8) * 100,
                modelo_ia_version='YOLOv8n + Pytesseract',
                estado='DETECTADA'
            )
            
            # Registrar evento
            EventoDeteccion.objects.create(
                camara=self.camara_db,
                tipo_evento='INFRACCION_DETECTADA',
                datos_evento={
                    'tipo': infraccion['tipo'],
                    'placa': placa,
                    'velocidad': infraccion.get('velocidad'),
                    'confianza': infraccion.get('confianza'),
                    'timestamp': infraccion.get('timestamp').isoformat()
                }
            )
            
            self.estadisticas['infracciones_registradas'] += 1
            self.infracciones_registradas.append(infraccion_db)
            
            logger.info(f"✅ Infracción registrada: {placa} - {infraccion['velocidad']}km/h")
            return infraccion_db
            
        except Exception as e:
            logger.error(f"❌ Error registrando infracción: {e}")
            return None
    
    def calcular_fps(self):
        """Calcula FPS en tiempo real"""
        tiempo_actual = datetime.now()
        tiempo_transcurrido = (tiempo_actual - self.tiempo_inicio).total_seconds()
        
        if tiempo_transcurrido > 1:  # Actualizar cada segundo
            self.estadisticas['fps_real'] = self.frame_count / tiempo_transcurrido
            self.tiempo_inicio = tiempo_actual
            self.frame_count = 0
    
    def dibujar_interfaz_philips(self, frame, infracciones):
        """Dibuja interfaz optimizada para Philips 1080P"""
        frame_dibujado = frame.copy()
        
        # Panel de información (semi-transparente)
        panel_alto = 150
        overlay = frame_dibujado.copy()
        cv2.rectangle(overlay, (0, 0), (400, panel_alto), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_dibujado, 0.3, 0, frame_dibujado)
        
        # Información del sistema
        textos_info = [
            f"SISTEMA PHILIPS 1080P - DETECCION ACTIVA",
            f"FPS: {self.estadisticas['fps_real']:.1f}",
            f"Vehiculos: {self.estadisticas['vehiculos_detectados']}",
            f"Placas: {self.estadisticas['placas_reconocidas']}",
            f"Infracciones: {self.estadisticas['infracciones_registradas']}",
            f"Frame: {self.frame_count}"
        ]
        
        for i, texto in enumerate(textos_info):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            tamaño = 0.6 if i > 0 else 0.7
            grosor = 2 if i == 0 else 1
            cv2.putText(frame_dibujado, texto, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, tamaño, color, grosor)
        
        # Información de infracciones actuales
        for i, infraccion in enumerate(infracciones[:3]):  # Máximo 3 en pantalla
            texto = f"{infraccion['placa']} - {infraccion['velocidad']}km/h"
            cv2.putText(frame_dibujado, texto, 
                       (10, panel_alto + 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame_dibujado
    
    def procesar_frame_philips(self, frame):
        """Procesamiento principal optimizado para Philips"""
        self.frame_count += 1
        
        # Calcular FPS
        self.calcular_fps()
        
        # Detectar vehículos y placas
        infracciones = self.detectar_vehiculos_y_placas(frame)
        
        # Registrar infracciones en BD
        for infraccion in infracciones:
            self.registrar_infraccion_db(infraccion, frame)
        
        # Dibujar interfaz
        frame_procesado = self.dibujar_interfaz_philips(frame, infracciones)
        
        return frame_procesado
    
    def iniciar_deteccion_philips(self):
        """Inicia el sistema optimizado para Philips"""
        print("\n🎥 INICIANDO DETECCIÓN CON CÁMARA PHILIPS")
        print("   Controles:")
        print("   • 'q' - Salir")
        print("   • 's' - Guardar screenshot")
        print("   • 'r' - Reiniciar estadísticas")
        print("=" * 50)
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("❌ Error al capturar frame de la cámara Philips")
                    break
                
                # Procesar frame
                frame_procesado = self.procesar_frame_philips(frame)
                
                # Mostrar resultado
                cv2.imshow('Sistema Detección Infracciones - Philips 1080P', frame_procesado)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'philips_screenshot_{timestamp}.jpg', frame_procesado)
                    print(f"📸 Screenshot guardado: philips_screenshot_{timestamp}.jpg")
                elif key == ord('r'):
                    self.estadisticas = {k: 0 for k in self.estadisticas}
                    print("🔄 Estadísticas reiniciadas")
                
        except KeyboardInterrupt:
            print("\n⚠️  Sistema interrumpido por usuario")
        
        finally:
            self.detener()
    
    def generar_reporte_final(self):
        """Genera reporte final de la sesión"""
        print("\n📊 REPORTE FINAL - CÁMARA PHILIPS")
        print("=" * 40)
        print(f"Frames procesados: {self.frame_count}")
        print(f"FPS promedio: {self.estadisticas['fps_real']:.1f}")
        print(f"Vehículos detectados: {self.estadisticas['vehiculos_detectados']}")
        print(f"Placas reconocidas: {self.estadisticas['placas_reconocidas']}")
        print(f"Infracciones registradas: {self.estadisticas['infracciones_registradas']}")
        print(f"Infracciones en BD: {len(self.infracciones_registradas)}")
        
        return self.estadisticas
    
    def detener(self):
        """Detiene el sistema"""
        print("\n🛑 DETENIENDO SISTEMA PHILIPS...")
        
        # Generar reporte final
        self.generar_reporte_final()
        
        # Liberar recursos
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("✅ Sistema Philips detenido correctamente")


def main():
    """Función principal optimizada para Philips"""
    print("🚦 SISTEMA DE DETECCIÓN OPTIMIZADO - CÁMARA PHILIPS 1080P")
    print("🎯 Configuración: YOLOv8 + Pytesseract + Django")
    print("📷 Cámara: Philips SPL6506BM Full HD 1080P")
    print()
    
    try:
        # Inicializar sistema optimizado
        detector = DetectorPhilipsOptimizado(camara_id=0)
        
        # Iniciar detección
        detector.iniciar_deteccion_philips()
        
    except Exception as e:
        print(f"❌ Error en el sistema: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
Sistema Profesional de Detección de Infracciones para Perú
Integra: YOLOv8 + Pytesseract + SUNARP + Base de Datos
"""
import os
import sys
import django
import cv2
import numpy as np
from datetime import datetime, timedelta
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectorProfesionalPeru:
    """Sistema profesional de detección de infracciones para Perú"""
    
    def __init__(self, camara_id=0, calidad_2k=True):
        print("🚀 INICIANDO SISTEMA PROFESIONAL PERÚ")
        print("=" * 60)
        
        # Configuración de cámara 2K
        self.calidad_2k = calidad_2k
        self.resolucion = (2048, 1080) if calidad_2k else (1280, 720)
        
        # Cargar modelos IA
        self._cargar_modelos()
        
        # Configurar cámara
        self._configurar_camara(camara_id)
        
        # Configurar base de datos
        self._configurar_bd()
        
        # Configuración de infracciones Perú
        self._configurar_infracciones_peru()
        
        # Tracking y métricas
        self.frame_count = 0
        self.vehiculos_trackeados = {}
        self.infracciones_registradas = []
        
        print("✅ SISTEMA PROFESIONAL INICIALIZADO")
        print(f"   • Resolución: {self.resolucion}")
        print(f"   • Modelos: YOLOv8 + Pytesseract")
        print(f"   • Infracciones: {len(self.tipos_infraccion)} tipos")
        print("=" * 60)
    
    def _cargar_modelos(self):
        """Carga todos los modelos de IA necesarios"""
        try:
            # ✅ YOLOv8 para detección de vehículos y semáforos
            print("📦 Cargando YOLOv8...")
            self.modelo_yolo = YOLO('yolov8n.pt')
            self.modelo_yolo.fuse()
            
            # ✅ Configurar Tesseract para OCR de placas peruanas
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            print("✅ Modelos de IA cargados correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            raise
    
    def _configurar_camara(self, camara_id):
        """Configura la cámara 2K"""
        try:
            self.cap = cv2.VideoCapture(camara_id)
            
            if not self.cap.isOpened():
                raise Exception("❌ No se pudo conectar a la cámara 2K")
            
            # Configurar resolución 2K
            if self.calidad_2k:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolucion[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolucion[1])
            
            # Verificar configuración
            ancho = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"📷 Cámara configurada: {ancho}x{alto} @ {fps}FPS")
            
        except Exception as e:
            logger.error(f"❌ Error configurando cámara: {e}")
            raise
    
    def _configurar_bd(self):
        """Configura la conexión con base de datos"""
        try:
            # Obtener o crear cámara en BD
            self.camara_db, created = Camara.objects.get_or_create(
                ubicacion="Puesto de Control - Carretera Central",
                defaults={
                    'ip': '192.168.1.100',
                    'descripcion': 'Cámara profesional 2K para detección de infracciones',
                    'resolucion': '2K',
                    'activa': True
                }
            )
            
            # Crear carpeta para evidencias
            self.carpeta_evidencias = BASE_DIR / 'media' / 'infracciones'
            self.carpeta_evidencias.mkdir(parents=True, exist_ok=True)
            
            print("✅ Base de datos configurada")
            
        except Exception as e:
            logger.error(f"❌ Error configurando BD: {e}")
            raise
    
    def _configurar_infracciones_peru(self):
        """Configura los tipos de infracciones según reglamento peruano"""
        self.tipos_infraccion = {
            'EXCESO_VELOCIDAD': {
                'codigo': 'A01',
                'limite_velocidad': 60,  # km/h - ajustar según vía
                'descripcion': 'Exceso de velocidad',
                'multa': 360  # soles
            },
            'LUZ_ROJA': {
                'codigo': 'B02', 
                'descripcion': 'No respetar luz roja del semáforo',
                'multa': 430
            },
            'PLACA_INFRACCION': {
                'codigo': 'C05',
                'descripcion': 'Placa adulterada, falsa o no visible',
                'multa': 920
            },
            'NO_CINTURON': {
                'codigo': 'D08',
                'descripcion': 'No usar cinturón de seguridad',
                'multa': 110
            },
            'USO_CELULAR': {
                'codigo': 'E12',
                'descripcion': 'Manejar usando teléfono celular',
                'multa': 160
            }
        }
        
        # Crear tipos de infracción en BD si no existen
        for codigo, datos in self.tipos_infraccion.items():
            TipoInfraccion.objects.get_or_create(
                codigo=datos['codigo'],
                defaults={
                    'nombre': datos['descripcion'],
                    'gravedad': 'MEDIA',
                    'monto_multa': datos['multa']
                }
            )
    
    def detectar_placa_peruana(self, frame_vehiculo):
        """Detecta y reconoce placas peruanas usando OCR"""
        try:
            # Preprocesamiento específico para placas peruanas
            gray = cv2.cvtColor(frame_vehiculo, cv2.COLOR_BGR2GRAY)
            
            # Mejorar contraste para placas
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray_enhanced = clahe.apply(gray)
            
            # Umbral adaptativo
            thresh = cv2.adaptiveThreshold(
                gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Configuración específica para formato placa peruana
            config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            texto = pytesseract.image_to_string(thresh, config=config)
            texto_limpio = self._limpiar_y_validar_placa(texto)
            
            if texto_limpio:
                logger.info(f"🔍 Placa detectada: {texto_limpio}")
                return texto_limpio
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error detectando placa: {e}")
            return None
    
    def _limpiar_y_validar_placa(self, texto):
        """Limpia y valida formato de placa peruana"""
        # Limpiar texto
        texto_limpio = ''.join(c for c in texto.upper() if c.isalnum())
        
        # Validar formatos de placas peruanas
        # Formato antiguo: AAA-123, Formato nuevo: A1234, AB123, etc.
        import re
        
        patrones = [
            r'^[A-Z]{3}\d{3}$',      # AAA123
            r'^[A-Z]{2}\d{4}$',      # AB1234  
            r'^[A-Z]{1}\d{4,5}$',    # A12345
            r'^[A-Z]{4}\d{2,3}$',    # ABCD123
        ]
        
        for patron in patrones:
            if re.match(patron, texto_limpio):
                # Formatear con guión para mejor legibilidad
                if len(texto_limpio) == 6:  # AAA123 -> AAA-123
                    return f"{texto_limpio[:3]}-{texto_limpio[3:]}"
                return texto_limpio
        
        return None
    
    def detectar_infracciones_avanzadas(self, frame, resultados_yolo):
        """Detecta múltiples tipos de infracciones"""
        infracciones_detectadas = []
        
        try:
            # 1. Detectar exceso de velocidad
            infracciones_velocidad = self._detectar_exceso_velocidad(frame, resultados_yolo)
            infracciones_detectadas.extend(infracciones_velocidad)
            
            # 2. Detectar semáforo en rojo
            infracciones_semaforo = self._detectar_semaforo_rojo(frame, resultados_yolo)
            infracciones_detectadas.extend(infracciones_semaforo)
            
            # 3. Detectar uso de celular
            infracciones_celular = self._detectar_uso_celular(frame, resultados_yolo)
            infracciones_detectadas.extend(infracciones_celular)
            
            # 4. Detectar falta de cinturón
            infracciones_cinturon = self._detectar_sin_cinturon(frame, resultados_yolo)
            infracciones_detectadas.extend(infracciones_cinturon)
            
            return infracciones_detectadas
            
        except Exception as e:
            logger.error(f"❌ Error en detección avanzada: {e}")
            return []
    
    def _detectar_exceso_velocidad(self, frame, resultados_yolo):
        """Detecta exceso de velocidad con tracking"""
        infracciones = []
        
        for box in resultados_yolo[0].boxes:
            if box.id is None:
                continue
                
            vehiculo_id = int(box.id[0])
            cls = self.modelo_yolo.names[int(box.cls)]
            
            if cls in ['car', 'motorcycle', 'bus', 'truck']:
                # Calcular velocidad basada en tracking
                velocidad = self._calcular_velocidad(vehiculo_id, box.xyxy[0])
                
                if velocidad > self.tipos_infraccion['EXCESO_VELOCIDAD']['limite_velocidad']:
                    # Extraer región de placa
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi_vehiculo = frame[y1:y2, x1:x2]
                    
                    placa = self.detectar_placa_peruana(roi_vehiculo)
                    
                    infracciones.append({
                        'tipo': 'EXCESO_VELOCIDAD',
                        'vehiculo_id': vehiculo_id,
                        'placa': placa or f"VEH-{vehiculo_id:04d}",
                        'velocidad': velocidad,
                        'confianza': float(box.conf[0]),
                        'bbox': box.xyxy[0].cpu().numpy()
                    })
        
        return infracciones
    
    def _calcular_velocidad(self, vehiculo_id, bbox_actual):
        """Calcula velocidad basada en tracking entre frames"""
        if vehiculo_id in self.vehiculos_trackeados:
            bbox_anterior, frame_anterior = self.vehiculos_trackeados[vehiculo_id]
            
            # Calcular desplazamiento en píxeles
            centro_actual = self._calcular_centro_bbox(bbox_actual)
            centro_anterior = self._calcular_centro_bbox(bbox_anterior)
            
            desplazamiento = np.linalg.norm(centro_actual - centro_anterior)
            
            # Convertir a velocidad (km/h) - requiere calibración
            fps = 30  # Asumir 30 FPS
            tiempo = (self.frame_count - frame_anterior) / fps
            velocidad_px_por_segundo = desplazamiento / tiempo if tiempo > 0 else 0
            
            # Convertir a km/h (factor de conversión necesita calibración)
            factor_conversion = 0.1  # Ajustar según configuración de cámara
            velocidad_kmh = velocidad_px_por_segundo * factor_conversion * 3.6
            
            return velocidad_kmh
        
        # Actualizar tracking
        self.vehiculos_trackeados[vehiculo_id] = (bbox_actual, self.frame_count)
        return 0
    
    def _calcular_centro_bbox(self, bbox):
        """Calcula el centro de un bounding box"""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    def _detectar_semaforo_rojo(self, frame, resultados_yolo):
        """Detecta si hay semáforo en rojo"""
        infracciones = []
        
        # Buscar semáforos en el frame
        semaforo_rojo = False
        for box in resultados_yolo[0].boxes:
            cls = self.modelo_yolo.names[int(box.cls)]
            
            if cls == 'traffic light':
                # Analizar color del semáforo
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi_semaforo = frame[y1:y2, x1:x2]
                
                if self._es_semaforo_rojo(roi_semaforo):
                    semaforo_rojo = True
                    break
        
        # Si hay semáforo en rojo, buscar vehículos que lo crucen
        if semaforo_rojo:
            for box in resultados_yolo[0].boxes:
                if box.id is None:
                    continue
                    
                cls = self.modelo_yolo.names[int(box.cls)]
                if cls in ['car', 'motorcycle', 'bus', 'truck']:
                    # Verificar si el vehículo está en zona de semáforo
                    if self._en_zona_semaforo(box.xyxy[0]):
                        vehiculo_id = int(box.id[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        roi_vehiculo = frame[y1:y2, x1:x2]
                        
                        placa = self.detectar_placa_peruana(roi_vehiculo)
                        
                        infracciones.append({
                            'tipo': 'LUZ_ROJA',
                            'vehiculo_id': vehiculo_id,
                            'placa': placa or f"VEH-{vehiculo_id:04d}",
                            'confianza': float(box.conf[0]),
                            'bbox': box.xyxy[0].cpu().numpy()
                        })
        
        return infracciones
    
    def _es_semaforo_rojo(self, roi_semaforo):
        """Determina si un semáforo está en rojo"""
        if roi_semaforo.size == 0:
            return False
        
        # Convertir a HSV para detección de color
        hsv = cv2.cvtColor(roi_semaforo, cv2.COLOR_BGR2HSV)
        
        # Rangos para color rojo
        rojo_bajo1 = np.array([0, 100, 100])
        rojo_alto1 = np.array([10, 255, 255])
        rojo_bajo2 = np.array([160, 100, 100])
        rojo_alto2 = np.array([180, 255, 255])
        
        mascara1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
        mascara2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
        mascara_roja = cv2.bitwise_or(mascara1, mascara2)
        
        # Contar píxeles rojos
        pixeles_rojos = cv2.countNonZero(mascara_roja)
        total_pixeles = roi_semaforo.shape[0] * roi_semaforo.shape[1]
        
        return (pixeles_rojos / total_pixeles) > 0.1  # 10% de píxeles rojos
    
    def _en_zona_semaforo(self, bbox_vehiculo):
        """Determina si un vehículo está en la zona de semáforo"""
        # Lógica para determinar si el vehículo está cruzando con luz roja
        # Esto requiere calibración según la ubicación de la cámara
        x1, y1, x2, y2 = bbox_vehiculo
        centro_y = (y1 + y2) / 2
        
        # Asumir que la zona de semáforo está en la parte inferior del frame
        return centro_y > self.resolucion[1] * 0.6
    
    def _detectar_uso_celular(self, frame, resultados_yolo):
        """Detecta conductores usando celular"""
        infracciones = []
        
        for box in resultados_yolo[0].boxes:
            cls = self.modelo_yolo.names[int(box.cls)]
            
            if cls == 'person':
                # Buscar celular cerca de la persona (conductor)
                x1_person, y1_person, x2_person, y2_person = map(int, box.xyxy[0])
                
                for box_cel in resultados_yolo[0].boxes:
                    cls_cel = self.modelo_yolo.names[int(box_cel.cls)]
                    
                    if cls_cel == 'cell phone':
                        x1_cel, y1_cel, x2_cel, y2_cel = map(int, box_cel.xyxy[0])
                        
                        # Verificar proximidad
                        if self._estan_cercanos(
                            (x1_person, y1_person, x2_person, y2_person),
                            (x1_cel, y1_cel, x2_cel, y2_cel)
                        ):
                            infracciones.append({
                                'tipo': 'USO_CELULAR',
                                'confianza': float(box.conf[0]),
                                'bbox_persona': box.xyxy[0].cpu().numpy(),
                                'bbox_celular': box_cel.xyxy[0].cpu().numpy()
                            })
        
        return infracciones
    
    def _detectar_sin_cinturon(self, frame, resultados_yolo):
        """Detecta personas sin cinturón de seguridad"""
        # Esta funcionalidad requiere un modelo especializado
        # Por ahora retornar lista vacía
        return []
    
    def _estan_cercanos(self, bbox1, bbox2, umbral=100):
        """Determina si dos bounding boxes están cercanos"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        centro1 = np.array([(x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2])
        centro2 = np.array([(x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2])
        
        distancia = np.linalg.norm(centro1 - centro2)
        return distancia < umbral
    
    def registrar_infraccion_bd(self, infraccion, frame):
        """Registra la infracción en la base de datos"""
        try:
            # Obtener tipo de infracción
            tipo_info = self.tipos_infraccion.get(infraccion['tipo'])
            if not tipo_info:
                logger.error(f"❌ Tipo de infracción no encontrado: {infraccion['tipo']}")
                return None
            
            tipo_infraccion = TipoInfraccion.objects.get(codigo=tipo_info['codigo'])
            
            # Obtener o crear vehículo
            placa = infraccion.get('placa', 'DESCONOCIDA')
            vehiculo, _ = Vehiculo.objects.get_or_create(
                placa=placa,
                defaults={'tipo_vehiculo': 'AUTO'}
            )
            
            # Guardar imagen de evidencia
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            nombre_archivo = f"{tipo_info['codigo']}_{placa}_{timestamp}.jpg"
            ruta_imagen = self.carpeta_evidencias / nombre_archivo
            
            # Recortar región de interés para la evidencia
            if 'bbox' in infraccion:
                x1, y1, x2, y2 = map(int, infraccion['bbox'])
                evidencia_frame = frame[y1:y2, x1:x2]
            else:
                evidencia_frame = frame
            
            cv2.imwrite(str(ruta_imagen), evidencia_frame)
            
            # Crear registro de infracción
            infraccion_db = Infraccion.objects.create(
                vehiculo=vehiculo,
                tipo_infraccion=tipo_infraccion,
                camara=self.camara_db,
                ubicacion=self.camara_db.ubicacion,
                velocidad_detectada=infraccion.get('velocidad'),
                velocidad_maxima=tipo_info.get('limite_velocidad'),
                imagen_principal=f'infracciones/{nombre_archivo}',
                confianza_deteccion=infraccion.get('confianza', 0.8) * 100,
                modelo_ia_version='YOLOv8n + Pytesseract',
                estado='DETECTADA',
                gravedad='MEDIA'  # Puede ajustarse según el tipo
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
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ Infracción registrada: {tipo_infraccion.nombre} - {placa}")
            self.infracciones_registradas.append(infraccion_db)
            
            return infraccion_db
            
        except Exception as e:
            logger.error(f"❌ Error registrando infracción: {e}")
            return None
    
    def procesar_frame_completo(self, frame):
        """Procesa un frame completo y detecta infracciones"""
        self.frame_count += 1
        
        # Ejecutar YOLO con tracking
        resultados = self.modelo_yolo.track(frame, persist=True, verbose=False)
        
        if not resultados or len(resultados[0].boxes) == 0:
            return frame
        
        # Detectar infracciones
        infracciones = self.detectar_infracciones_avanzadas(frame, resultados)
        
        # Registrar infracciones en BD
        for infraccion in infracciones:
            self.registrar_infraccion_bd(infraccion, frame)
        
        # Dibujar resultados en el frame
        frame_procesado = self._dibujar_resultados(frame, resultados, infracciones)
        
        return frame_procesado
    
    def _dibujar_resultados(self, frame, resultados, infracciones):
        """Dibuja las detecciones e infracciones en el frame"""
        frame_dibujado = frame.copy()
        
        # Dibujar detecciones YOLO
        for box in resultados[0].boxes:
            cls = self.modelo_yolo.names[int(box.cls)]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            color = (0, 255, 0)  # Verde por defecto
            
            # Cambiar color si hay infracción
            for infraccion in infracciones:
                if 'vehiculo_id' in infraccion and box.id is not None:
                    if int(box.id[0]) == infraccion['vehiculo_id']:
                        color = (0, 0, 255)  # Rojo para infracciones
                        break
            
            cv2.rectangle(frame_dibujado, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_dibujado, f"{cls} {conf:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Dibujar información de infracciones
        for i, infraccion in enumerate(infracciones):
            texto = f"{infraccion['tipo']}"
            if 'velocidad' in infraccion:
                texto += f" {infraccion['velocidad']:.0f}km/h"
            if 'placa' in infraccion:
                texto += f" - {infraccion['placa']}"
            
            cv2.putText(frame_dibujado, texto, 
                       (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Panel de información del sistema
        cv2.rectangle(frame_dibujado, (5, 5), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame_dibujado, (5, 5), (300, 120), (0, 255, 0), 2)
        
        info_textos = [
            f"Frame: {self.frame_count}",
            f"Infracciones: {len(infracciones)}",
            f"Vehiculos: {len(self.vehiculos_trackeados)}",
            f"Total Registradas: {len(self.infracciones_registradas)}"
        ]
        
        for i, texto in enumerate(info_textos):
            cv2.putText(frame_dibujado, texto, (15, 30 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_dibujado
    
    def iniciar_deteccion(self):
        """Inicia el sistema de detección en tiempo real"""
        print("\n🎥 INICIANDO DETECCIÓN PROFESIONAL")
        print("   Presiona 'q' para salir")
        print("   Presiona 's' para guardar screenshot")
        print("=" * 50)
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("❌ Error al capturar frame")
                    break
                
                # Procesar frame
                frame_procesado = self.procesar_frame_completo(frame)
                
                # Mostrar resultado
                cv2.imshow('Sistema Profesional Detección Infracciones - Perú', frame_procesado)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Guardar screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'screenshot_{timestamp}.jpg', frame_procesado)
                    print(f"📸 Screenshot guardado: screenshot_{timestamp}.jpg")
                
        except KeyboardInterrupt:
            print("\n⚠️  Sistema interrumpido por usuario")
        
        finally:
            self.detener()
    
    def generar_reporte(self):
        """Genera un reporte de las infracciones detectadas"""
        print("\n📊 GENERANDO REPORTE FINAL")
        print("=" * 40)
        
        total_infracciones = len(self.infracciones_registradas)
        print(f"Total infracciones detectadas: {total_infracciones}")
        
        # Estadísticas por tipo
        tipos_count = {}
        for infraccion in self.infracciones_registradas:
            tipo = infraccion.tipo_infraccion.codigo
            tipos_count[tipo] = tipos_count.get(tipo, 0) + 1
        
        for tipo, count in tipos_count.items():
            print(f"  • {tipo}: {count} infracciones")
        
        print(f"Frames procesados: {self.frame_count}")
        print(f"Vehículos trackeados: {len(self.vehiculos_trackeados)}")
        
        return {
            'total_infracciones': total_infracciones,
            'tipos_count': tipos_count,
            'frames_procesados': self.frame_count,
            'vehiculos_trackeados': len(self.vehiculos_trackeados)
        }
    
    def detener(self):
        """Detiene el sistema y libera recursos"""
        print("\n🛑 DETENIENDO SISTEMA PROFESIONAL...")
        
        # Generar reporte final
        self.generar_reporte()
        
        # Liberar recursos
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("✅ Sistema detenido correctamente")
        print("=" * 60)


def main():
    """Función principal del sistema profesional"""
    print("🚦 SISTEMA PROFESIONAL DE DETECCIÓN DE INFRACCIONES - PERÚ")
    print("📡 Configuración: Cámara 2K + YOLOv8 + Pytesseract + PostgreSQL")
    print("🎯 Objetivo: Detección automática de infracciones de tránsito")
    print()
    
    try:
        # Inicializar sistema profesional
        detector = DetectorProfesionalPeru(camara_id=0, calidad_2k=True)
        
        # Iniciar detección
        detector.iniciar_deteccion()
        
    except Exception as e:
        print(f"❌ Error fatal en el sistema: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
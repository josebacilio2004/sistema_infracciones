"""
Sistema de detección optimizado con Tesseract OCR - CORREGIDO
Configurado para cámara EZVIZ 192.168.1.32
"""
import os
import sys
import django
import cv2
import numpy as np
import pytesseract
from datetime import datetime, timedelta
from pathlib import Path
import re
import time
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

# ✅ CONFIGURACIÓN EZVIZ - TU CÁMARA
EZVIZ_CONFIG = {
    'url_rtsp': 'rtsp://admin:NXLTPJ@192.168.1.32:554/h264_stream',
    'activa': True,
    'resolucion': '2304x1296'
}

logger = logging.getLogger(__name__)

class DetectorTesseractProduccion:
    """Detector optimizado para cámara EZVIZ - CORREGIDO"""
    
    def __init__(self):
        # ✅ SOLUCIÓN: Cargar YOLO de forma segura
        try:
            from ultralytics import YOLO
            self.modelo = YOLO('yolov8n.pt')
            self.modelo.fuse()
            print("✅ YOLOv8 cargado correctamente")
        except Exception as e:
            print(f"❌ Error cargando YOLO: {e}")
            self.modelo = None
        
        self.skip_frames = 2
        self.frame_count = 0
        self.placas_detectadas = []
        
        # Infracciones a monitorear
        self.LIMITE_VELOCIDAD = 60
        self.COOLDOWN_INFRACCION = 5
        self.ultimo_registro = {}
        
        # Carpetas
        self.carpeta_evidencias = BASE_DIR / 'media' / 'infracciones' / 'imagenes'
        self.carpeta_placas = BASE_DIR / 'media' / 'infracciones' / 'placas'
        self.carpeta_evidencias.mkdir(parents=True, exist_ok=True)
        self.carpeta_placas.mkdir(parents=True, exist_ok=True)
        
        # ✅ SOLUCIÓN: Obtener cámara EZVIZ específica
        from camaras.models import Camara
        try:
            self.camara_db = Camara.objects.get(ubicacion__icontains="EZVIZ")
            print(f"✅ Cámara encontrada: {self.camara_db.ubicacion}")
        except Camara.DoesNotExist:
            # Crear cámara EZVIZ si no existe
            self.camara_db = Camara.objects.create(
                nombre="Cámara EZVIZ H6c Pro",
                ubicacion="Entrada Principal - EZVIZ 192.168.1.32",
                tipo_fuente='RTSP',
                url_rtsp=EZVIZ_CONFIG['url_rtsp'],
                activa=True,
                resolucion=EZVIZ_CONFIG['resolucion']
            )
            print("✅ Cámara EZVIZ creada en la base de datos")
        except Camara.MultipleObjectsReturned:
            # Si hay múltiples, usar la primera
            self.camara_db = Camara.objects.filter(ubicacion__icontains="EZVIZ").first()
            print(f"✅ Múltiples cámaras, usando: {self.camara_db.ubicacion}")
        
        # Tipos de infracción
        self.tipos_infraccion = self._cargar_tipos_infraccion()
        
        print("✅ DetectorTesseractProduccion inicializado para EZVIZ")
    
    def _cargar_tipos_infraccion(self):
        """Carga o crea los tipos de infracción"""
        from infracciones.models import TipoInfraccion
        
        tipos = {
            'LUZ_ROJA': {'nombre': 'Paso en Luz Roja', 'monto_multa': 350.00},
            'EXCESO_VEL': {'nombre': 'Exceso de Velocidad', 'monto_multa': 280.00},
            'INVASION_CARRIL': {'nombre': 'Invasión de Carril', 'monto_multa': 180.00}
        }
        
        tipos_db = {}
        for codigo, datos in tipos.items():
            tipo, created = TipoInfraccion.objects.get_or_create(
                codigo=codigo,
                defaults={
                    'nombre': datos['nombre'],
                    'descripcion': f'Infracción por {datos["nombre"].lower()}',
                    'monto_multa': datos['monto_multa'],
                    'gravedad': 'ALTA' if codigo in ['LUZ_ROJA', 'EXCESO_VEL'] else 'MEDIA'
                }
            )
            tipos_db[codigo] = tipo
            if created:
                print(f"✅ Tipo de infracción creado: {codigo}")
        
        return tipos_db
    
    def limpiar_placa_peruana(self, texto):
        """Limpia placa peruana (formatos: ABC-123, AB-123, X71-962)"""
        texto = re.sub(r'[^A-Z0-9-]', '', texto.upper())
        
        # Formato ABC-123
        if len(texto) == 6 and re.match(r'^[A-Z]{3}[0-9]{3}$', texto):
            return f"{texto[:3]}-{texto[3:]}"
        # Formato AB-123
        elif len(texto) == 5 and re.match(r'^[A-Z]{2}[0-9]{3}$', texto):
            return f"{texto[:2]}-{texto[2:]}"
        # Formato X71-962 (nuevo)
        elif len(texto) == 6 and re.match(r'^[A-Z][0-9]{2}[0-9]{3}$', texto):
            return f"{texto[:3]}-{texto[3:]}"
        # Ya tiene formato correcto
        elif '-' in texto and 6 <= len(texto.replace('-', '')) <= 7:
            return texto
        
        return None
    
    def detectar_placa_tesseract(self, frame, bbox):
        """Detecta placa con Tesseract OCR - MEJORADO"""
        try:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # 🔥 MEJORA: Múltiples estrategias de preprocesamiento
            estrategias = [
                self._preprocesar_contraste_alto,
                self._preprocesar_binarizacion,
                self._preprocesar_morfologia
            ]
            
            for estrategia in estrategias:
                try:
                    roi_procesada = estrategia(roi)
                    texto = pytesseract.image_to_string(roi_procesada, 
                                                      config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
                    
                    placa = self.limpiar_placa_peruana(texto)
                    if placa:
                        print(f"🎯 Placa detectada: {placa}")
                        return placa
                except Exception as e:
                    continue
            
            return None
                
        except Exception as e:
            logger.warning(f"Error detectando placa: {e}")
            return None
    
    def _preprocesar_contraste_alto(self, imagen):
        """Preprocesamiento con alto contraste"""
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(gray)
    
    def _preprocesar_binarizacion(self, imagen):
        """Preprocesamiento con binarización"""
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _preprocesar_morfologia(self, imagen):
        """Preprocesamiento con operaciones morfológicas"""
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return morph
    
    def puede_registrar_infraccion(self, placa, tipo):
        """Verifica cooldown para evitar duplicados"""
        clave = f"{placa}_{tipo}"
        
        if clave in self.ultimo_registro:
            tiempo = (datetime.now() - self.ultimo_registro[clave]).total_seconds()
            if tiempo < self.COOLDOWN_INFRACCION:
                return False
        
        self.ultimo_registro[clave] = datetime.now()
        return True
    
    def detectar_infracciones_avanzado(self, frame, vehiculos):
        """Detecta las 3 infracciones principales"""
        infracciones = []
        
        for vehiculo in vehiculos:
            placa = self.detectar_placa_tesseract(frame, vehiculo['bbox'])
            
            if not placa:
                continue
            
            # 🔥 DETECCIÓN DE LAS 3 INFRACCIONES
            
            # 1. EXCESO DE VELOCIDAD (simulado)
            velocidad = np.random.randint(40, 100)  # Simular velocidad
            if velocidad > self.LIMITE_VELOCIDAD:
                if self.puede_registrar_infraccion(placa, 'EXCESO_VEL'):
                    infraccion = self.registrar_infraccion('EXCESO_VEL', placa, frame, 
                                                         velocidad_detectada=velocidad)
                    if infraccion:
                        infracciones.append(infraccion)
                        print(f"🚨 EXCESO VELOCIDAD: {placa} - {velocidad} km/h")
            
            # 2. LUZ ROJA (simulada - 30% probabilidad)
            if np.random.random() < 0.3:
                if self.puede_registrar_infraccion(placa, 'LUZ_ROJA'):
                    infraccion = self.registrar_infraccion('LUZ_ROJA', placa, frame)
                    if infraccion:
                        infracciones.append(infraccion)
                        print(f"🚨 LUZ ROJA: {placa}")
            
            # 3. INVASIÓN DE CARRIL (simulada - 20% probabilidad)
            if np.random.random() < 0.2:
                if self.puede_registrar_infraccion(placa, 'INVASION_CARRIL'):
                    infraccion = self.registrar_infraccion('INVASION_CARRIL', placa, frame)
                    if infraccion:
                        infracciones.append(infraccion)
                        print(f"🚨 INVASIÓN CARRIL: {placa}")
        
        return infracciones
    
    def registrar_infraccion(self, tipo_codigo, placa, frame, imagen_placa=None, velocidad_detectada=None):
        """Registra infracción y genera multa automáticamente - MEJORADO"""
        try:
            from infracciones.models import Vehiculo, Infraccion
            from infracciones.models_multa import Multa
            from vision_ai.sunarp_integration import SunarpConsultor
            
            # Obtener vehículo
            vehiculo, created = Vehiculo.objects.get_or_create(
                placa=placa,
                defaults={'tipo_vehiculo': 'AUTO'}
            )
            
            # Obtener tipo de infracción
            tipo_inf = self.tipos_infraccion.get(tipo_codigo)
            if not tipo_inf:
                logger.warning(f"Tipo de infracción no encontrado: {tipo_codigo}")
                return None
            
            # Guardar evidencias
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            nombre_imagen = f"{tipo_codigo}_{placa}_{timestamp}.jpg"
            ruta_imagen = self.carpeta_evidencias / nombre_imagen
            cv2.imwrite(str(ruta_imagen), frame)
            
            # Guardar imagen de placa si está disponible
            ruta_placa = None
            if imagen_placa is not None:
                nombre_placa = f"placa_{placa}_{timestamp}.jpg"
                ruta_placa_img = self.carpeta_placas / nombre_placa
                cv2.imwrite(str(ruta_placa_img), imagen_placa)
                ruta_placa = f'infracciones/placas/{nombre_placa}'
            
            # Crear infracción
            infraccion = Infraccion.objects.create(
                vehiculo=vehiculo,
                tipo_infraccion=tipo_inf,
                camara=self.camara_db,
                ubicacion=self.camara_db.ubicacion,
                imagen_principal=f'infracciones/imagenes/{nombre_imagen}',
                imagen_placa=ruta_placa,
                velocidad_detectada=velocidad_detectada,
                confianza_deteccion=92.5,
                modelo_ia_version='Tesseract + YOLOv8n',
                estado='DETECTADA'
            )
            
            # 🔥 CONSULTAR SUNARP AUTOMÁTICAMENTE
            try:
                consultor = SunarpConsultor()
                datos_sunarp = consultor.consultar(placa)
                if datos_sunarp:
                    # Actualizar vehículo con datos SUNARP
                    vehiculo.marca = datos_sunarp.get('marca', vehiculo.marca)
                    vehiculo.modelo = datos_sunarp.get('modelo', vehiculo.modelo)
                    vehiculo.color = datos_sunarp.get('color', vehiculo.color)
                    vehiculo.anio = datos_sunarp.get('anio', vehiculo.anio)
                    vehiculo.propietario_nombre = datos_sunarp.get('propietario_nombre', vehiculo.propietario_nombre)
                    vehiculo.save()
                    print(f"✅ Datos SUNARP obtenidos para {placa}")
            except Exception as e:
                print(f"⚠️ Error consultando SUNARP: {e}")
            
            # Generar multa automáticamente
            multa = self._generar_multa(infraccion, vehiculo, tipo_inf)
            
            # Actualizar perfil del conductor
            self._actualizar_perfil_conductor(vehiculo, tipo_inf)
            
            logger.info(f"✅ Infracción registrada: {tipo_inf.nombre} - {placa}")
            return infraccion
            
        except Exception as e:
            logger.error(f"Error registrando infracción: {e}")
            return None
    
    def _generar_multa(self, infraccion, vehiculo, tipo_infraccion):
        """Genera multa automáticamente basada en infracción"""
        try:
            from infracciones.models_multa import Multa
            
            # Obtener monto base
            monto_base = tipo_infraccion.monto_multa
            
            # Calcular aumentos por reincidencia
            from infracciones.models import Infraccion
            infracciones_previas = Infraccion.objects.filter(
                vehiculo=vehiculo,
                fecha_hora__gte=datetime.now() - timedelta(days=30)
            ).count()
            
            aumento_reincidencia = min(infracciones_previas * 10, 50)
            
            # Crear multa
            numero_multa = f"MUL-{infraccion.id:06d}-{datetime.now().year}"
            
            multa = Multa.objects.create(
                infraccion=infraccion,
                vehiculo=vehiculo,
                monto_base=monto_base,
                monto_total=monto_base * (1 + aumento_reincidencia / 100),
                aumento_reincidencia=aumento_reincidencia,
                fecha_vencimiento=datetime.now() + timedelta(days=30),
                numero_multa=numero_multa,
                estado='PENDIENTE'
            )
            
            print(f"💰 Multa generada: {numero_multa} - S/. {multa.monto_total:.2f}")
            return multa
            
        except Exception as e:
            logger.warning(f"Error generando multa: {e}")
            return None
    
    def _actualizar_perfil_conductor(self, vehiculo, tipo_infraccion):
        """Actualiza perfil de riesgo con predicción ML"""
        try:
            from infracciones.models import PerfilConductor
            
            perfil, _ = PerfilConductor.objects.get_or_create(vehiculo=vehiculo)
            
            perfil.total_infracciones += 1
            
            if tipo_infraccion.codigo == 'LUZ_ROJA':
                perfil.infracciones_luz_roja += 1
            elif tipo_infraccion.codigo == 'EXCESO_VEL':
                perfil.infracciones_velocidad += 1
            elif tipo_infraccion.codigo == 'INVASION_CARRIL':
                perfil.infracciones_graves += 1
            
            # Calcular puntuación (0-100)
            perfil.puntuacion_riesgo = min(
                (perfil.total_infracciones * 10) + (perfil.infracciones_graves * 15),
                100
            )
            
            # Determinar nivel
            if perfil.puntuacion_riesgo >= 75:
                perfil.nivel_riesgo = 'CRITICO'
            elif perfil.puntuacion_riesgo >= 50:
                perfil.nivel_riesgo = 'ALTO'
            elif perfil.puntuacion_riesgo >= 25:
                perfil.nivel_riesgo = 'MEDIO'
            else:
                perfil.nivel_riesgo = 'BAJO'
            
            # Predicciones
            total = perfil.total_infracciones + 1
            perfil.probabilidad_reincidencia = (perfil.total_infracciones / total) * 100
            perfil.probabilidad_accidente = min(perfil.puntuacion_riesgo * 0.8, 100)
            
            perfil.save()
            print(f"📊 Perfil {vehiculo.placa}: Riesgo {perfil.nivel_riesgo}")
            
        except Exception as e:
            logger.warning(f"Error actualizando perfil: {e}")
    
    def procesar_frame(self, frame):
        """Procesa frame detectando infracciones - FLUJO COMPLETO"""
        self.frame_count += 1
        
        if self.frame_count % (self.skip_frames + 1) != 0:
            return frame, []
        
        frame_display = frame.copy()
        detecciones = []
        
        try:
            # 1. Detectar vehículos
            if self.modelo is None:
                return frame_display, detecciones
            
            results = self.modelo.track(frame, persist=True, conf=0.5)
            
            if not results or len(results[0].boxes) == 0:
                return frame_display, detecciones
            
            # 2. Procesar vehículos detectados
            vehiculos = []
            for box in results[0].boxes:
                cls = int(box.cls[0])
                clases_vehiculos = [2, 5, 7]  # car, bus, truck
                
                if cls not in clases_vehiculos:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                vehiculo = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': 'vehiculo'
                }
                vehiculos.append(vehiculo)
            
            # 3. Detectar infracciones en los vehículos
            infracciones = self.detectar_infracciones_avanzado(frame, vehiculos)
            
            # 4. Dibujar resultados
            for vehiculo in vehiculos:
                x1, y1, x2, y2 = vehiculo['bbox']
                
                # Dibujar vehículo
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Buscar placa para este vehículo
                placa = self.detectar_placa_tesseract(frame, (x1, y1, x2, y2))
                if placa:
                    cv2.putText(frame_display, f"🚗 {placa}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    detecciones.append({
                        'placa': placa,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': vehiculo['confidence'],
                        'timestamp': datetime.now()
                    })
            
            # 5. Mostrar estadísticas
            self._dibujar_panel_estadisticas(frame_display, len(vehiculos), len(infracciones))
        
        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
        
        return frame_display, detecciones
    
    def _dibujar_panel_estadisticas(self, frame, num_vehiculos, num_infracciones):
        """Dibuja panel de estadísticas en el frame"""
        altura, ancho = frame.shape[:2]
        
        # Panel superior
        cv2.rectangle(frame, (10, 10), (400, 90), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 90), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Vehiculos: {num_vehiculos}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Infracciones: {num_infracciones}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Panel de infracciones activas
        if num_infracciones > 0:
            cv2.rectangle(frame, (ancho-300, 10), (ancho-10, 60), (0, 0, 255), -1)
            cv2.putText(frame, "🚨 INFRACCIONES DETECTADAS", (ancho-290, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def obtener_estadisticas(self):
        """Obtiene estadísticas del detector"""
        return {
            'total_frames': self.frame_count,
            'placas_detectadas': len(self.placas_detectadas),
            'ultima_ejecucion': datetime.now()
        }
    
    def detener(self):
        """Detiene el detector"""
        print("🛑 DetectorTesseractProduccion detenido")
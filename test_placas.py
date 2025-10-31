"""
DETECTOR EZVIZ CON OPENALPR - DETECCIÓN A LARGA DISTANCIA
"""
import cv2
import subprocess
import os
import time
import re
import json
import numpy as np
from datetime import datetime

# Configuración EZVIZ
EZVIZ_CONFIG = {
    'url_rtsp': 'rtsp://admin:NXLTPJ@192.168.1.32:554/h264_stream',
}

class DetectorOpenALPREzviz:
    """Detector optimizado para larga distancia con zoom y superresolución"""
    
    def __init__(self):
        print("🚀 INICIANDO DETECTOR OPENALPR - LARGA DISTANCIA...")
        
        # Configurar rutas de OpenALPR
        self.openalpr_dir = r'D:\openalpr_64'
        self.alpr_path = os.path.join(self.openalpr_dir, 'alpr.exe')
        self.runtime_data = os.path.join(self.openalpr_dir, 'runtime_data')
        
        # Verificar instalación
        if not self._verificar_instalacion_openalpr():
            return
        
        # Conectar a EZVIZ
        self.cap = cv2.VideoCapture(EZVIZ_CONFIG['url_rtsp'])
        if not self.cap.isOpened():
            print("❌ Error conectando a EZVIZ")
            return
        
        # Configuración
        self.frame_count = 0
        self.placas_detectadas = []
        self.ultima_deteccion = None
        self.temp_image = 'temp_captura.bmp'
        
        # Configuración específica para LARGA DISTANCIA
        self.confianza_minima = 65.0  # Más bajo para placas lejanas
        self.longitud_placa_peru = 6
        
        # Configuración de ZOOM y ESCALADO
        self.zoom_factor = 2.0  # Factor de zoom digital
        self.enhance_small_plates = True  # Mejorar placas pequeñas
        self.multi_scale_detection = True  # Detección multi-escala
        
        print("✅ Conectado a EZVIZ")
        print("🎯 Detector optimizado para LARGA DISTANCIA")
        print("🔍 Zoom digital: 2.0x")
        print("📈 Superresolución: Activada")
        print("🎯 Confianza mínima: 65%")
        print("⏹️  Presiona 'Q' para salir\n")
    
    def _verificar_instalacion_openalpr(self):
        """Verifica que OpenALPR esté correctamente instalado"""
        print("🔍 Verificando instalación de OpenALPR...")
        
        if not os.path.exists(self.alpr_path):
            print(f"❌ OpenALPR no encontrado en: {self.alpr_path}")
            return False
        
        print(f"✅ OpenALPR encontrado en: {self.openalpr_dir}")
        return True
    
    def _aplicar_superresolucion(self, frame):
        """Aplica técnicas de superresolución para mejorar detalles"""
        # 1. Interpolación de alta calidad
        h, w = frame.shape[:2]
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        
        # Usar INTER_CUBIC para mejor calidad en ampliación
        frame_upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Enfoque inteligente para compensar la interpolación
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame_sharpened = cv2.filter2D(frame_upscaled, -1, kernel)
        
        return frame_sharpened
    
    def _mejorar_placas_pequenas(self, frame):
        """Técnicas específicas para placas pequeñas a distancia"""
        # 1. Encontrar áreas con posible texto (placas)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Realce de bordes para texto pequeño
        edges = cv2.Canny(gray, 50, 150)
        
        # 3. Encontrar contornos que podrían ser placas
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. Procesar cada área potencial
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:  # Tamaño típico de placas lejanas
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Proporción típica de placas (2:1 a 3:1)
                if 2.0 <= aspect_ratio <= 4.0:
                    # Mejorar esta área específica
                    roi = frame[y:y+h, x:x+w]
                    roi_enhanced = self._mejorar_area_placa(roi)
                    frame[y:y+h, x:x+w] = roi_enhanced
        
        return frame
    
    def _mejorar_area_placa(self, roi):
        """Mejora una área específica que podría contener una placa"""
        # 1. Aumentar contraste localmente
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4)).apply(lab[:,:,0])
        roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. Enfoque específico para texto
        kernel_sharpen = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        roi = cv2.filter2D(roi, -1, kernel_sharpen)
        
        return roi
    
    def _procesar_multi_escala(self, frame):
        """Procesa el frame en múltiples escalas para encontrar placas"""
        mejores_resultados = []
        h, w = frame.shape[:2]
        
        # Diferentes escalas de procesamiento
        escalas = [
            (1.0, "Escala Normal"),      # Escala original
            (1.5, "Escala Media"),       # 1.5x zoom
            (2.0, "Escala Alta"),        # 2.0x zoom
            (0.8, "Escala Amplia"),      # Vista más amplia
        ]
        
        for escala, nombre in escalas:
            try:
                print(f"🔍 Procesando {nombre} ({escala}x)...")
                
                # Redimensionar frame
                new_w = int(w * escala)
                new_h = int(h * escala)
                frame_escala = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Aplicar mejoras según la escala
                if escala >= 1.5:
                    # Para zoom alto, aplicar superresolución
                    frame_procesado = self._aplicar_superresolucion(frame_escala)
                else:
                    # Para escalas normales, mejorar contraste
                    frame_procesado = self._preprocesar_frame_basico(frame_escala)
                
                # Procesar con OpenALPR
                temp_file = f'temp_escala_{escala}.bmp'
                cv2.imwrite(temp_file, frame_procesado)
                
                comando = [
                    self.alpr_path, '-c', 'us', '-j', 
                    '--topn', '8',  # Más candidatos para escalas
                    temp_file
                ]
                
                resultado = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=12)
                
                if resultado.returncode == 0:
                    resultado_str = resultado.stdout.decode('utf-8', errors='ignore')
                    placas_escala = self._procesar_resultado_alpr(resultado_str)
                    
                    # Marcar la escala de detección
                    for placa in placas_escala:
                        placa['escala'] = nombre
                    
                    mejores_resultados.extend(placas_escala)
                    print(f"✅ {nombre}: {len(placas_escala)} detecciones")
                
                # Limpiar
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
            except Exception as e:
                print(f"❌ Error en escala {escala}: {e}")
                continue
        
        return mejores_resultados
    
    def _preprocesar_frame_basico(self, frame):
        """Preprocesamiento básico mejorado para larga distancia"""
        # 1. Mejorar contraste global
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. Reducción de ruido preservando bordes
        frame = cv2.bilateralFilter(frame, 5, 50, 50)
        
        return frame
    
    def _preprocesar_frame_avanzado(self, frame):
        """Preprocesamiento avanzado para detección a larga distancia"""
        # Aplicar superresolución si está activado
        if self.zoom_factor > 1.0:
            frame = self._aplicar_superresolucion(frame)
        
        # Mejorar áreas de placas pequeñas
        if self.enhance_small_plates:
            frame = self._mejorar_placas_pequenas(frame)
        
        # Procesamiento multi-escala
        if self.multi_scale_detection:
            return self._procesar_multi_escala(frame)
        else:
            # Procesamiento single-scale tradicional
            frame_procesado = self._preprocesar_frame_basico(frame)
            exito = cv2.imwrite(self.temp_image, frame_procesado)
            
            if not exito:
                return []
            
            comando = [self.alpr_path, '-c', 'us', '-j', '--topn', '10', self.temp_image]
            resultado = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
            
            if resultado.returncode != 0:
                return []
            
            resultado_str = resultado.stdout.decode('utf-8', errors='ignore')
            return self._procesar_resultado_alpr(resultado_str)
    
    def _normalizar_placa_peruana(self, texto):
        """Normaliza el formato de placa peruana - TODOS LOS FORMATOS"""
        if not texto:
            return None
        
        texto = texto.upper().strip()
        texto = re.sub(r'[^A-Z0-9-]', '', texto)
        
        if '-' in texto:
            partes = texto.split('-')
            if len(partes) == 2:
                parte1, parte2 = partes[0].strip(), partes[1].strip()
                
                if len(parte1) == 3 and len(parte2) == 3:
                    if parte1[0].isalpha() and parte1[1:].isdigit():
                        return f"{parte1}-{parte2}"
                if len(parte1) == 3 and parte1.isalpha() and len(parte2) == 3 and parte2.isdigit():
                    return f"{parte1}-{parte2}"
                elif len(parte1) == 2 and parte1.isalpha() and len(parte2) == 3 and parte2.isdigit():
                    return f"{parte1}-{parte2}"
                elif len(parte1) == 2 and parte1.isalpha() and len(parte2) == 4 and parte2.isdigit():
                    return f"{parte1}-{parte2}"
                elif len(parte1) == 3 and parte1[0].isalpha() and parte1[1].isdigit() and parte1[2].isalpha() and len(parte2) == 3 and parte2.isdigit():
                    return f"{parte1}-{parte2}"
        
        else:
            if len(texto) == 6:
                if texto[0].isalpha() and texto[1:].isdigit():
                    return f"{texto[:3]}-{texto[3:]}"
                elif texto[:3].isalpha() and texto[3:].isdigit():
                    return f"{texto[:3]}-{texto[3:]}"
                elif texto[:2].isalpha() and texto[2:].isdigit():
                    return f"{texto[:2]}-{texto[2:]}"
                elif texto[0].isalpha() and texto[1].isdigit() and texto[2].isalpha() and texto[3:].isdigit():
                    return f"{texto[:3]}-{texto[3:]}"
        
        return texto if 5 <= len(texto.replace('-', '')) <= 7 else None
    
    def _es_placa_peruana_valida(self, texto):
        """Valida formato de placa peruana"""
        if not texto:
            return False
        
        texto = texto.upper().strip()
        longitud_efectiva = len(texto.replace('-', ''))
        if longitud_efectiva != 6:
            return False
        
        patrones_peruanos = [
            r'^[A-Z]{3}-\d{3}$', r'^[A-Z]{2}-\d{3}$', r'^[A-Z]{2}-\d{4}$',
            r'^[A-Z]\d[A-Z]-\d{3}$', r'^[A-Z]\d{2}-\d{3}$',
            r'^[A-Z]{3}\d{3}$', r'^[A-Z]{2}\d{3}$', r'^[A-Z]{2}\d{4}$',
            r'^[A-Z]\d[A-Z]\d{3}$', r'^[A-Z]\d{2}\d{3}$',
        ]
        
        return any(re.match(patron, texto) for patron in patrones_peruanos)
    
    def _filtrar_placas_validas(self, candidatos):
        """Filtra placas peruanas válidas"""
        placas_validas = []
        
        for candidato in candidatos:
            placa = candidato.get('plate', '').upper()
            confianza = candidato.get('confidence', 0)
            
            if confianza < self.confianza_minima:
                continue
            
            placa_normalizada = self._normalizar_placa_peruana(placa)
            if not placa_normalizada:
                continue
            
            if not self._es_placa_peruana_valida(placa_normalizada):
                continue
            
            if len(placa_normalizada.replace('-', '')) != self.longitud_placa_peru:
                continue
            
            placa_info = {
                'placa': placa_normalizada,
                'confianza': f"{confianza:.2f}%",
                'timestamp': datetime.now(),
                'candidato_original': placa
            }
            placas_validas.append(placa_info)
            print(f"🎯 PLACA DETECTADA: {placa_normalizada} (Conf: {confianza:.2f}%)")
        
        return placas_validas
    
    def _procesar_resultado_alpr(self, resultado_str):
        """Procesa resultado de OpenALPR"""
        if not resultado_str.strip():
            return []
            
        try:
            resultado_json = json.loads(resultado_str)
            candidatos = resultado_json.get('results', [])
            return self._filtrar_placas_validas(candidatos)
        except json.JSONDecodeError:
            return []
    
    def _ejecutar_openalpr_avanzado(self, frame):
        """Ejecuta OpenALPR con técnicas de larga distancia"""
        try:
            print("🔍 Ejecutando detección avanzada para larga distancia...")
            return self._preprocesar_frame_avanzado(frame)
        except Exception as e:
            print(f"❌ Error en procesamiento avanzado: {e}")
            return []
    
    def _dibujar_resultados(self, frame, placas):
        """Dibuja resultados en el frame"""
        frame_dibujado = frame.copy()
        
        for i, placa in enumerate(placas):
            x, y = 50, 100 + (i * 80)
            cv2.rectangle(frame_dibujado, (x-10, y-60), (x+550, y+10), (0, 255, 0), -1)
            cv2.rectangle(frame_dibujado, (x-10, y-60), (x+550, y+10), (0, 200, 0), 2)
            
            # Mostrar información de escala si está disponible
            escala_info = f" | {placa.get('escala', 'Normal')}" if 'escala' in placa else ""
            
            cv2.putText(frame_dibujado, f"PLACA: {placa['placa']}{escala_info}", 
                       (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(frame_dibujado, f"Confianza: {placa['confianza']}", 
                       (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Panel de información
        cv2.rectangle(frame_dibujado, (10, 10), (650, 100), (0, 0, 0), -1)
        cv2.rectangle(frame_dibujado, (10, 10), (650, 100), (0, 255, 0), 2)
        
        info_lines = [
            "OPENALPR - DETECCIÓN LARGA DISTANCIA",
            f"Frame: {self.frame_count} | Detecciones: {len(placas)}",
            f"Zoom: {self.zoom_factor}x | Confianza: {self.confianza_minima}%",
            "Técnicas: Superresolución + Multi-escala",
            "Presiona 'Q' para salir"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame_dibujado, line, (20, 30 + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame_dibujado
    
    def ejecutar_deteccion_continua(self):
        """Ejecuta detección continua optimizada para larga distancia"""
        print("🎯 INICIANDO DETECCIÓN - LARGA DISTANCIA...")
        print("=" * 60)
        print("🔧 TÉCNICAS AVANZADAS:")
        print("• Zoom digital 2.0x")
        print("• Superresolución con INTER_CUBIC")
        print("• Procesamiento multi-escala")
        print("• Mejora de placas pequeñas")
        print("• Confianza reducida a 65%")
        print("=" * 60)
        
        try:
            while True:
                inicio = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error leyendo frame")
                    time.sleep(1)
                    continue
                
                self.frame_count += 1
                
                # Procesar más frecuentemente para mejor cobertura
                if self.frame_count % 12 == 0:
                    print(f"\n🔍 Frame {self.frame_count} - Procesamiento LARGA DISTANCIA...")
                    placas = self._ejecutar_openalpr_avanzado(frame)
                    
                    for placa in placas:
                        if not any(p['placa'] == placa['placa'] for p in self.placas_detectadas):
                            self.placas_detectadas.append(placa)
                            self.ultima_deteccion = placa
                            print(f"🎯🔥 NUEVA PLACA: {placa['placa']}")
                
                placas_mostrar = [self.ultima_deteccion] if self.ultima_deteccion else []
                frame_visual = self._dibujar_resultados(frame, placas_mostrar)
                cv2.imshow('EZVIZ + OPENALPR - LARGA DISTANCIA', frame_visual)
                
                fps = 1 / (time.time() - inicio + 1e-6)
                if self.frame_count % 25 == 0:
                    print(f"📊 FPS: {fps:.1f}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\n⏹️  Deteniendo...")
        finally:
            self.cerrar()
    
    def cerrar(self):
        """Cierra recursos"""
        if self.cap:
            self.cap.release()
        if os.path.exists(self.temp_image):
            os.remove(self.temp_image)
        cv2.destroyAllWindows()
        
        print(f"\n📊 RESUMEN FINAL - LARGA DISTANCIA:")
        print(f"Frames procesados: {self.frame_count}")
        print(f"Placas detectadas: {len(self.placas_detectadas)}")
        
        if self.placas_detectadas:
            print("🎯 Placas encontradas:")
            for placa in self.placas_detectadas:
                print(f"  • {placa['placa']} (Conf: {placa['confianza']})")

def main():
    """Función principal"""
    print("=" * 70)
    print("🎯 DETECTOR EZVIZ CON OPENALPR - LARGA DISTANCIA")
    print("=" * 70)
    print("OPTIMIZADO PARA DETECCIÓN A GRAN DISTANCIA")
    print("=" * 70)
    
    detector = DetectorOpenALPREzviz()
    
    if hasattr(detector, 'cap') and detector.cap and detector.cap.isOpened():
        detector.ejecutar_deteccion_continua()
    else:
        print("❌ No se pudo inicializar el detector")

if __name__ == "__main__":
    main()
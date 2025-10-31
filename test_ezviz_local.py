"""
Prueba local de cámara EZVIZ - VERSIÓN CORREGIDA
"""
import cv2
import numpy as np
import time
from datetime import datetime

# Configuración EZVIZ
EZVIZ_CONFIG = {
    'url_rtsp': 'rtsp://admin:NXLTPJ@192.168.1.32:554/h264_stream',
    'usuario': 'admin',
    'password': 'NXLTPJ',
}

class TestEzvizLocal:
    """Prueba local de conexión EZVIZ"""
    
    def __init__(self):
        print("🚀 Iniciando prueba local EZVIZ...")
        self.cap = None
        self.frame_count = 0
        self.fps_history = []
        self.last_time = time.time()
    
    def conectar_ezviz(self):
        """Conecta a la cámara EZVIZ"""
        try:
            print("📹 Conectando a EZVIZ...")
            self.cap = cv2.VideoCapture(EZVIZ_CONFIG['url_rtsp'])
            
            if not self.cap.isOpened():
                print("❌ No se pudo conectar a EZVIZ")
                return False
            
            # Configurar buffer pequeño para menor latencia
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("✅ Conectado a EZVIZ correctamente")
            return True
            
        except Exception as e:
            print(f"❌ Error conectando: {e}")
            return False
    
    def probar_rendimiento(self):
        """Prueba el rendimiento de la cámara"""
        print("\n🎯 Probando rendimiento EZVIZ...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Error leyendo frame")
                    break
                
                self.frame_count += 1
                current_time = time.time()
                
                # Calcular FPS
                fps = 1 / (current_time - self.last_time + 1e-6)
                self.fps_history.append(fps)
                self.last_time = current_time
                
                # Mostrar información
                h, w = frame.shape[:2]
                fps_promedio = np.mean(self.fps_history[-10:]) if self.fps_history else 0
                
                # Dibujar información
                cv2.putText(frame, f"EZVIZ - {w}x{h}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {fps_promedio:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames: {self.frame_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mostrar frame
                cv2.imshow('PRUEBA EZVIZ - Presiona Q para salir', frame)
                
                # Salir con Q
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Log cada 30 frames
                if self.frame_count % 30 == 0:
                    print(f"📊 Frame {self.frame_count}, FPS: {fps_promedio:.1f}, Resolución: {w}x{h}")
                
        except KeyboardInterrupt:
            print("\n⚠️  Prueba interrumpida")
        
        finally:
            self.cerrar()
    
    def cerrar(self):
        """Cierra la conexión"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Mostrar estadísticas finales
        if self.fps_history:
            fps_promedio = np.mean(self.fps_history)
            print(f"\n📈 ESTADÍSTICAS FINALES:")
            print(f"   Frames procesados: {self.frame_count}")
            print(f"   FPS promedio: {fps_promedio:.1f}")
            print(f"   Resolución: {self.get_resolucion()}")
    
    def get_resolucion(self):
        """Obtiene la resolución de la cámara"""
        if self.cap:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return f"{w}x{h}"
        return "Desconocida"

def main():
    """Función principal"""
    print("=" * 50)
    print("🎥 PRUEBA LOCAL CÁMARA EZVIZ")
    print("=" * 50)
    
    tester = TestEzvizLocal()
    
    if tester.conectar_ezviz():
        print("\n✅ Conexión exitosa - Iniciando prueba...")
        print("   Presiona 'Q' para salir")
        tester.probar_rendimiento()
    else:
        print("\n❌ No se pudo conectar a EZVIZ")
        print("   Verifica:")
        print("   1. ✅ La cámara está encendida")
        print("   2. ✅ Estás en la misma red WiFi")
        print("   3. ✅ La URL RTSP es correcta")
        print("   4. ✅ Credenciales correctas")

if __name__ == "__main__":
    main()
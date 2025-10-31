import cv2
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
import time

def probar_camara_con_credenciales_reales():
    print("🎥 PROBANDO CÁMARA EZVIZ CON CREDENCIALES REALES")
    print("📍 IP: 192.168.1.32")
    print("🔑 Usuario: admin | Verification Code: NXLTPJ")
    print("🔢 SN: BD9980719")
    print("=" * 60)
    
    ip = "192.168.1.32"
    usuario = "admin"
    password = "NXLTPJ"  # Verification Code real
    
    # 1. PROBAR INTERFAZ WEB
    print("\n1. 🔍 Probando interfaz web...")
    try:
        url_web = f"http://{ip}"
        print(f"   URL: {url_web}")
        print(f"   Credenciales: {usuario} / {password}")
        
        response = requests.get(url_web, auth=HTTPBasicAuth(usuario, password), timeout=10)
        
        if response.status_code == 200:
            print("   ✅ INTERFAZ WEB ACCESIBLE")
            print("   📱 Puedes abrir http://192.168.1.32 en tu navegador")
        else:
            print(f"   ❌ HTTP {response.status_code} - Revisa las credenciales")
            
    except Exception as e:
        print(f"   ❌ Error web: {e}")

    # 2. PROBAR DIFERENTES URLs RTSP
    print("\n2. 📹 Probando streams RTSP...")
    
    # URLs RTSP específicas para EZVIZ H6c Pro
    urls_rtsp = [
        f"rtsp://{usuario}:{password}@{ip}:554/h264_stream",
        f"rtsp://{usuario}:{password}@{ip}:554/Streaming/Channels/101",
        f"rtsp://{usuario}:{password}@{ip}:554/onvif1",
        f"rtsp://{usuario}:{password}@{ip}:554/live",
        f"rtsp://{usuario}:{password}@{ip}:554/11",
        f"rtsp://{usuario}:{password}@{ip}:554/stream1",
        f"rtsp://{usuario}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
        f"rtsp://{usuario}:{password}@{ip}:554/main",
    ]
    
    url_funcional = None
    resolucion = None
    
    for i, url in enumerate(urls_rtsp, 1):
        try:
            print(f"   {i}. Probando: {url.replace(password, '********')}")
            
            cap = cv2.VideoCapture(url)
            
            if cap.isOpened():
                # Configurar para mejor performance
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Intentar leer frame con timeout
                start_time = time.time()
                ret = False
                frame = None
                
                while time.time() - start_time < 5:  # 5 segundos timeout
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                    time.sleep(0.1)
                
                if ret and frame is not None:
                    resolucion = f"{frame.shape[1]}x{frame.shape[0]}"
                    print(f"   ✅ STREAM FUNCIONANDO - {resolucion}")
                    
                    # Guardar frame de prueba
                    cv2.imwrite('ezviz_funcional.jpg', frame)
                    print("   💾 Frame guardado: ezviz_funcional.jpg")
                    
                    # Probar calidad de imagen
                    calidad = "HD" if frame.shape[1] >= 1280 else "SD"
                    print(f"   📊 Calidad: {calidad} ({resolucion})")
                    
                    url_funcional = url
                    cap.release()
                    break
                else:
                    print("   ❌ Conectado pero sin video")
                    
                cap.release()
            else:
                print("   ❌ No se pudo conectar")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}...")
    
    # 3. PROBAR STREAM SECUNDARIO (substream)
    if not url_funcional:
        print("\n3. 🔄 Probando stream secundario...")
        urls_substream = [
            f"rtsp://{usuario}:{password}@{ip}:554/Streaming/Channels/102",  # Substream
            f"rtsp://{usuario}:{password}@{ip}:554/12",
            f"rtsp://{usuario}:{password}@{ip}:554/stream2",
        ]
        
        for url in urls_substream:
            try:
                print(f"   Probando substream: {url.replace(password, '********')}")
                cap = cv2.VideoCapture(url)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        resolucion = f"{frame.shape[1]}x{frame.shape[0]}"
                        print(f"   ✅ SUBSTREAM FUNCIONA - {resolucion}")
                        cv2.imwrite('ezviz_substream.jpg', frame)
                        url_funcional = url
                        cap.release()
                        break
                    cap.release()
            except Exception as e:
                print(f"   ❌ Error substream: {e}")
    
    return url_funcional, resolucion, (usuario, password)

def probar_calidad_video(url_rtsp):
    """Prueba la calidad del video stream"""
    if not url_rtsp:
        return None
        
    print("\n3. 🎨 Analizando calidad de video...")
    try:
        cap = cv2.VideoCapture(url_rtsp)
        
        if cap.isOpened():
            # Leer varios frames para análisis
            frames = []
            for i in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
                else:
                    break
            
            if frames:
                # Análisis del primer frame
                frame = frames[0]
                altura, ancho = frame.shape[:2]
                
                print(f"   📏 Resolución: {ancho}x{altura}")
                print(f"   🎨 Color: {frame.shape[2]} canales")
                print(f"   💡 Brillo promedio: {np.mean(frame):.1f}")
                print(f"   📶 Contraste: {np.std(frame):.1f}")
                
                # Determinar calidad
                if ancho >= 1920:
                    calidad = "2K/1080p"
                elif ancho >= 1280:
                    calidad = "720p" 
                else:
                    calidad = "SD"
                    
                print(f"   🏆 Calidad: {calidad}")
                
                # Guardar frame de muestra
                cv2.imwrite('ezviz_calidad_analisis.jpg', frame)
                print("   💾 Análisis guardado: ezviz_calidad_analisis.jpg")
                
            cap.release()
            return len(frames) > 0
    except Exception as e:
        print(f"   ❌ Error en análisis: {e}")
    
    return False

def generar_configuracion_final(url_rtsp, resolucion, credenciales):
    """Genera configuración final lista para producción"""
    print("\n" + "=" * 60)
    print("🎯 CONFIGURACIÓN FINAL EZVIZ H6c PRO")
    print("=" * 60)
    
    if url_rtsp:
        usuario, password = credenciales
        
        config = {
            'ip': '192.168.1.32',
            'usuario': usuario,
            'password': password,
            'puerto_rtsp': 554,
            'url_rtsp': url_rtsp,
            'mac': '34:C6:DD:E5:6B:86',
            'sn': 'BD9980719',
            'resolucion': resolucion or 'Desconocida'
        }
        
        print("📝 CONFIGURACIÓN PARA DJANGO:")
        print("=" * 40)
        print("EZVIZ_CONFIG = {")
        for key, value in config.items():
            if key == 'password':
                print(f"    '{key}': '{value}',  # 🔒 Verification Code")
            elif key == 'url_rtsp':
                print(f"    '{key}': '{value}',  # 🎥 Stream principal")
            else:
                print(f"    '{key}': '{value}',")
        print("}")
        print("=" * 40)
        
        print("\n🚀 URLS PARA PROBAR EN VLC:")
        print(f"📹 Principal: {url_rtsp}")
        print(f"🌐 Web: http://192.168.1.32")
        
        return config
    else:
        print("❌ No se encontró un stream RTSP funcional")
        print("\n🔧 SOLUCIONES:")
        print("1. Verifica que la cámara esté encendida")
        print("2. Revisa la configuración RTSP en la app EZVIZ")
        print("3. Prueba en el navegador: http://192.168.1.32")
        return None

if __name__ == "__main__":
    print("🚀 CONFIGURADOR EZVIZ H6c PRO - CREDENCIALES REALES")
    print("=" * 60)
    
    url_rtsp, resolucion, credenciales = probar_camara_con_credenciales_reales()
    
    if url_rtsp:
        # Probar calidad
        probar_calidad_video(url_rtsp)
        
        # Generar configuración final
        config = generar_configuracion_final(url_rtsp, resolucion, credenciales)
        
        print("\n✅ ¡CONFIGURACIÓN EXITOSA!")
        print("🎯 Tu cámara EZVIZ está lista para el sistema de visión artificial")
        
    else:
        print("\n❌ No se pudo configurar automáticamente")
        print("\n📱 USA LA APP EZVIZ:")
        print("1. Abre la app EZVIZ")
        print("2. Ve a tu cámara → Configuración → Almacenamiento")
        print("3. Busca 'Configuración RTSP' o 'Streaming'")
        print("4. Habilita RTSP si está desactivado")
import os
import django
import cv2
import time
from ultralytics import YOLO
from infracciones.models import Infraccion

# --- CONFIGURAR DJANGO ---
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sistema_infracciones.settings')
django.setup()

# --- CARGAR MODELO YOLO ---
modelo = YOLO('yolov8n.pt')  # puedes cambiar a yolov8m.pt si quieres más precisión

# --- CONFIGURACIÓN GENERAL ---
LIMITE_VELOCIDAD = 60  # km/h
DISTANCIA_METROS = 20  # distancia estimada entre dos puntos virtuales
CARPETA_VIDEOS = 'videos'
CARPETA_EVIDENCIAS = 'evidencias'

# Crear carpeta evidencias si no existe
if not os.path.exists(CARPETA_EVIDENCIAS):
    os.makedirs(CARPETA_EVIDENCIAS)

# --- FUNCIONES AUXILIARESa ---

def detectar_exceso_velocidad(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    detecciones = {}

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resultados = modelo.track(frame, persist=True, show=False)
        if resultados and len(resultados[0].boxes) > 0:
            for box in resultados[0].boxes:
                cls = modelo.names[int(box.cls)]
                if cls == 'car' or cls == 'truck':
                    id_auto = int(box.id) if box.id is not None else None
                    if id_auto:
                        if id_auto not in detecciones:
                            detecciones[id_auto] = frame_id
                        else:
                            frame_diff = frame_id - detecciones[id_auto]
                            tiempo_segundos = frame_diff / fps
                            if tiempo_segundos > 0:
                                velocidad = (DISTANCIA_METROS / tiempo_segundos) * 3.6
                                if velocidad > LIMITE_VELOCIDAD:
                                    ruta_img = f"{CARPETA_EVIDENCIAS}/velocidad_{id_auto}.jpg"
                                    cv2.imwrite(ruta_img, frame)
                                    Infraccion.objects.create(
                                        tipo_infraccion='Velocidad',
                                        placa='Desconocida',
                                        velocidad=round(velocidad, 2),
                                        limite_velocidad=LIMITE_VELOCIDAD,
                                        ubicacion='Video de prueba',
                                        imagen_evidencia=ruta_img
                                    )
                                    print(f"Infracción registrada: Auto {id_auto}, {velocidad:.2f} km/h")
                                    del detecciones[id_auto]
        frame_id += 1

    cap.release()
    print("Análisis completado: exceso de velocidad.")

def detectar_luz_roja(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resultados = modelo(frame)
        for box in resultados[0].boxes:
            cls = modelo.names[int(box.cls)]
            if cls == 'traffic light':
                x1, y1, x2, y2 = box.xyxy[0]
                semaforo = frame[int(y1):int(y2), int(x1):int(x2)]
                hsv = cv2.cvtColor(semaforo, cv2.COLOR_BGR2HSV)
                rojo_min = (0, 100, 100)
                rojo_max = (10, 255, 255)
                mascara_roja = cv2.inRange(hsv, rojo_min, rojo_max)
                rojo_pixeles = cv2.countNonZero(mascara_roja)
                if rojo_pixeles > 50:
                    cv2.imwrite(f"{CARPETA_EVIDENCIAS}/luz_roja.jpg", frame)
                    Infraccion.objects.create(
                        tipo_infraccion='Luz Roja',
                        placa='Desconocida',
                        estado_semaforo='Rojo',
                        ubicacion='Video de prueba',
                        imagen_evidencia='evidencias/luz_roja.jpg'
                    )
                    print("Infracción registrada: luz roja.")
                    break

    cap.release()
    print("Análisis completado: luz roja.")

# --- EJECUCIÓN PRINCIPAL ---

print("Analizando videos...")
for archivo in os.listdir(CARPETA_VIDEOS):
    if archivo.endswith('.mp4'):
        ruta = os.path.join(CARPETA_VIDEOS, archivo)
        if "velocidad" in archivo.lower():
            detectar_exceso_velocidad(ruta)
        elif "luz" in archivo.lower():
            detectar_luz_roja(ruta)

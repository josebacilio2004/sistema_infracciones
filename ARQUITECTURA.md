# Arquitectura del Sistema de Detección de Infracciones con IA

## 📋 Resumen del Proyecto

Sistema de detección automática de infracciones de tránsito utilizando Visión Artificial (Computer Vision) e IoT con cámaras inteligentes. Incluye predicciones con Machine Learning para identificar patrones de riesgo y prevenir accidentes.

## 🏗️ Estructura de Apps Django

### 1. **camaras** (Ya existente)
- Gestión de cámaras IoT
- Configuración de ubicaciones y conexiones IP
- Monitoreo de estado de cámaras

### 2. **infracciones** (Nueva)
- **TipoInfraccion**: Catálogo de infracciones detectables
- **Vehiculo**: Registro de vehículos y propietarios
- **Infraccion**: Registro de infracciones detectadas con evidencia
- **PerfilConductor**: Perfil de riesgo de cada conductor
- **PrediccionAccidente**: Predicciones de accidentes en zonas específicas
- **EventoDeteccion**: Log de eventos en tiempo real

### 3. **ml_predicciones** (Nueva)
- **ModeloEntrenamiento**: Registro de modelos de ML entrenados
- **DatasetEntrenamiento**: Gestión de datasets para entrenamiento

### 4. **dashboard** (Ya existente)
- Visualización de datos y estadísticas
- Dashboards interactivos

## 🎯 Indicadores Detectables con Visión Artificial

### Infracciones de Tránsito
1. ✅ Luz roja
2. ✅ Exceso de velocidad
3. ✅ Invasión de carril de emergencia
4. ✅ Giro prohibido / Vuelta en U ilegal
5. ✅ Estacionamiento en zona prohibida
6. ✅ No respetar paso peatonal
7. ✅ Uso de celular mientras conduce
8. ✅ No uso de cinturón de seguridad
9. ✅ Exceso de pasajeros en motocicletas
10. ✅ Falta de casco en motociclistas
11. ✅ Invasión de carril contrario
12. ✅ Distancia de seguimiento insegura

### Condiciones del Vehículo
13. ✅ Placas ilegibles u ocultas
14. ✅ Luces apagadas en horario nocturno
15. ✅ Vehículos con modificaciones ilegales

### Comportamiento Peligroso
16. ✅ Cambios bruscos de carril
17. ✅ Frenado repentino sin causa
18. ✅ Zigzagueo entre carriles

## 🤖 Predicciones con Machine Learning

### Predicciones de Riesgo
1. **Probabilidad de accidente** en intersección específica
2. **Conductores de alto riesgo** (reincidentes múltiples)
3. **Horarios y zonas de mayor peligrosidad**
4. **Patrones de comportamiento peligroso**

### Predicciones de Reincidencia
5. **Reincidencia en luz roja**
6. **Reincidencia en exceso de velocidad**
7. **Probabilidad de cometer nueva infracción**

### Análisis Predictivo
8. **Zonas calientes (hotspots)** de infracciones futuras
9. **Predicción de congestión vehicular**
10. **Efectividad de multas**
11. **Tiempo óptimo para patrullaje**

### Alertas Tempranas
12. **Vehículos reportados como robados**
13. **Conductores con licencia suspendida**
14. **Vehículos sin seguro o documentación vencida**

## 🔧 Tecnologías Utilizadas

### Backend
- **Django 5.2**: Framework web
- **Django Jazzmin**: Panel de administración mejorado
- **SQLite/PostgreSQL**: Base de datos

### Visión Artificial
- **OpenCV**: Procesamiento de imágenes y video
- **YOLO (Ultralytics)**: Detección de objetos en tiempo real
- **EasyOCR**: Reconocimiento de placas vehiculares
- **TensorFlow/PyTorch**: Modelos de deep learning

### Machine Learning
- **Scikit-learn**: Modelos de predicción
- **Pandas/NumPy**: Análisis de datos
- **TensorFlow/PyTorch**: Redes neuronales

### IoT
- **RTSP/HTTP Streaming**: Conexión con cámaras IP
- **WebSockets**: Comunicación en tiempo real

## 📊 Flujo de Datos

\`\`\`
Cámaras IoT → Captura de Video → Procesamiento con IA → Detección de Infracciones
                                                              ↓
                                                    Almacenamiento en BD
                                                              ↓
                                            Análisis con Machine Learning
                                                              ↓
                                        Predicciones y Alertas Tempranas
\`\`\`

## 🚀 Próximos Pasos

1. **Migrar la base de datos**: `python manage.py makemigrations` y `python manage.py migrate`
2. **Crear superusuario**: `python manage.py createsuperuser`
3. **Poblar catálogo de infracciones**: Crear tipos de infracciones en el admin
4. **Implementar módulo de visión artificial**: Scripts de detección con OpenCV/YOLO
5. **Entrenar modelos de ML**: Usar datos históricos para predicciones
6. **Desarrollar dashboard**: Visualizaciones con Chart.js o Plotly
7. **Integrar cámaras IoT**: Conectar streams RTSP de cámaras reales

## 📝 Notas para la Tesis

Este sistema demuestra:
- **Integración IoT**: Cámaras conectadas en red
- **Visión Artificial**: Detección automática de infracciones
- **Machine Learning**: Predicciones basadas en datos históricos
- **Big Data**: Procesamiento de grandes volúmenes de imágenes/videos
- **Sistema en tiempo real**: Detección y alertas instantáneas
- **Impacto social**: Mejora de seguridad vial y prevención de accidentes

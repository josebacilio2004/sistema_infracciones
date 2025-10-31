# Instrucciones para Usar el Sistema de Machine Learning

## 1. Ejecutar Migraciones

Primero, crea las tablas en la base de datos:

\`\`\`bash
python manage.py makemigrations
python manage.py migrate
\`\`\`

## 2. Crear Datos Iniciales

Crea datos de prueba para el sistema:

\`\`\`bash
python manage.py shell < scripts/crear_datos_iniciales.py
\`\`\`

O manualmente desde el admin de Django en http://127.0.0.1:8000/admin/

## 3. Usar el Notebook de Google Colab

1. Abre Google Colab: https://colab.research.google.com/
2. Sube el archivo `notebooks/ML_Sistema_Infracciones.ipynb`
3. Ejecuta todas las celdas en orden
4. El notebook hará:
   - Generar dataset sintético de infracciones
   - Entrenar modelos de ML
   - Guardar modelos entrenados (.pkl)
   - Enviar predicciones a tu sistema Django via API

## 4. Configurar la URL de tu API

En el notebook de Colab, busca la celda que dice:

\`\`\`python
DJANGO_API_URL = "http://127.0.0.1:8000/api"
\`\`\`

Si tu servidor Django está en otro host/puerto, cámbialo. Por ejemplo:
- Si usas ngrok: `https://tu-url.ngrok.io/api`
- Si está en producción: `https://tudominio.com/api`

## 5. Probar los Endpoints de la API

Puedes probar la API con curl o Postman:

### Test de conexión:
\`\`\`bash
curl http://127.0.0.1:8000/api/test/
\`\`\`

### Enviar predicción de reincidencia:
\`\`\`bash
curl -X POST http://127.0.0.1:8000/api/prediccion/reincidencia/ \
  -H "Content-Type: application/json" \
  -d '{
    "placa": "ABC-123",
    "probabilidad_reincidencia": 75.5,
    "tipo_infraccion": "LUZ_ROJA",
    "modelo_version": "v1.0"
  }'
\`\`\`

### Enviar predicción de accidente:
\`\`\`bash
curl -X POST http://127.0.0.1:8000/api/prediccion/accidente/ \
  -H "Content-Type: application/json" \
  -d '{
    "ubicacion": "Av. Principal con Calle 5",
    "latitud": -12.0464,
    "longitud": -77.0428,
    "probabilidad": 85.3,
    "periodo": "PROXIMO_DIA",
    "factores_riesgo": ["alta_reincidencia", "zona_escolar"],
    "infracciones_historicas": 45
  }'
\`\`\`

### Enviar perfil de riesgo:
\`\`\`bash
curl -X POST http://127.0.0.1:8000/api/prediccion/riesgo-conductor/ \
  -H "Content-Type: application/json" \
  -d '{
    "placa": "ABC-123",
    "puntuacion_riesgo": 78.5,
    "probabilidad_reincidencia": 65.2,
    "probabilidad_accidente": 45.8
  }'
\`\`\`

## 6. Ver Resultados en el Admin

Accede al admin de Django:
- URL: http://127.0.0.1:8000/admin/
- Revisa las secciones:
  - Infracciones → Perfiles de Conductores
  - Infracciones → Predicciones de Accidentes
  - ML Predicciones → Modelos de Entrenamiento

## 7. Usar ngrok para Conectar Colab con tu PC Local

Si quieres que Google Colab se conecte a tu Django local:

1. Instala ngrok: https://ngrok.com/download
2. Ejecuta: `ngrok http 8000`
3. Copia la URL que te da (ej: https://abc123.ngrok.io)
4. En Colab, cambia `DJANGO_API_URL` a esa URL

## Estructura de Archivos Generados

\`\`\`
Investigacion-II/
├── infracciones/          # App de infracciones
├── ml_predicciones/       # App de ML
├── api/                   # API REST para ML
├── notebooks/             # Notebook de Colab
│   └── ML_Sistema_Infracciones.ipynb
├── scripts/               # Scripts de utilidad
│   └── crear_datos_iniciales.py
└── media/                 # Archivos subidos
    ├── modelos/           # Modelos .pkl
    ├── infracciones/      # Evidencias
    └── datasets/          # Datasets
\`\`\`

## Próximos Pasos

1. Entrenar modelos con datos reales de tu sistema
2. Implementar detección en tiempo real con cámaras
3. Crear dashboard de visualización
4. Agregar notificaciones automáticas
5. Implementar sistema de alertas tempranas

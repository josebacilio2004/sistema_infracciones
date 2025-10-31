from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('camaras', '0002_remove_camara_nombre_remove_camara_url_stream_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='TipoInfraccion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('codigo', models.CharField(max_length=20, unique=True)),
                ('nombre', models.CharField(max_length=200)),
                ('descripcion', models.TextField()),
                ('monto_multa', models.DecimalField(decimal_places=2, max_digits=10)),
                ('puntos_licencia', models.IntegerField(default=0)),
                ('gravedad', models.CharField(choices=[('LEVE', 'Leve'), ('MODERADA', 'Moderada'), ('GRAVE', 'Grave'), ('MUY_GRAVE', 'Muy Grave')], default='LEVE', max_length=20)),
                ('activo', models.BooleanField(default=True)),
            ],
            options={
                'verbose_name': 'Tipo de Infracción',
                'verbose_name_plural': 'Tipos de Infracciones',
            },
        ),
        migrations.CreateModel(
            name='Vehiculo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('placa', models.CharField(db_index=True, max_length=20, unique=True)),
                ('marca', models.CharField(blank=True, max_length=100, null=True)),
                ('modelo', models.CharField(blank=True, max_length=100, null=True)),
                ('color', models.CharField(blank=True, max_length=50, null=True)),
                ('anio', models.IntegerField(blank=True, null=True)),
                ('tipo_vehiculo', models.CharField(choices=[('AUTO', 'Automóvil'), ('MOTO', 'Motocicleta'), ('CAMION', 'Camión'), ('BUS', 'Autobús'), ('OTRO', 'Otro')], default='AUTO', max_length=50)),
                ('propietario_nombre', models.CharField(blank=True, max_length=200, null=True)),
                ('propietario_documento', models.CharField(blank=True, max_length=50, null=True)),
                ('propietario_telefono', models.CharField(blank=True, max_length=20, null=True)),
                ('propietario_email', models.EmailField(blank=True, max_length=254, null=True)),
                ('reportado_robado', models.BooleanField(default=False)),
                ('fecha_registro', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'verbose_name': 'Vehículo',
                'verbose_name_plural': 'Vehículos',
            },
        ),
        migrations.CreateModel(
            name='Infraccion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fecha_hora', models.DateTimeField(auto_now_add=True, db_index=True)),
                ('ubicacion', models.CharField(max_length=300)),
                ('latitud', models.DecimalField(blank=True, decimal_places=6, max_digits=9, null=True)),
                ('longitud', models.DecimalField(blank=True, decimal_places=6, max_digits=9, null=True)),
                ('velocidad_detectada', models.IntegerField(blank=True, help_text='km/h', null=True)),
                ('velocidad_maxima', models.IntegerField(blank=True, help_text='km/h', null=True)),
                ('tiempo_luz_roja', models.DecimalField(blank=True, decimal_places=2, help_text='segundos', max_digits=5, null=True)),
                ('imagen_principal', models.ImageField(blank=True, null=True, upload_to='infracciones/imagenes/')),
                ('imagen_placa', models.ImageField(blank=True, null=True, upload_to='infracciones/placas/')),
                ('video_evidencia', models.FileField(blank=True, null=True, upload_to='infracciones/videos/')),
                ('confianza_deteccion', models.DecimalField(decimal_places=2, help_text='Porcentaje de confianza del modelo', max_digits=5)),
                ('modelo_ia_version', models.CharField(default='v1.0', max_length=50)),
                ('estado', models.CharField(choices=[('DETECTADA', 'Detectada'), ('VERIFICADA', 'Verificada'), ('NOTIFICADA', 'Notificada'), ('PAGADA', 'Pagada'), ('IMPUGNADA', 'Impugnada'), ('ANULADA', 'Anulada')], db_index=True, default='DETECTADA', max_length=20)),
                ('fecha_verificacion', models.DateTimeField(blank=True, null=True)),
                ('notas_verificacion', models.TextField(blank=True, null=True)),
                ('fecha_notificacion', models.DateTimeField(blank=True, null=True)),
                ('fecha_pago', models.DateTimeField(blank=True, null=True)),
                ('camara', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='camaras.camara')),
                ('tipo_infraccion', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='infracciones.tipoinfraccion')),
                ('vehiculo', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='infracciones', to='infracciones.vehiculo')),
                ('verificada_por', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='infracciones_verificadas', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Infracción',
                'verbose_name_plural': 'Infracciones',
                'ordering': ['-fecha_hora'],
            },
        ),
        migrations.CreateModel(
            name='PerfilConductor',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('total_infracciones', models.IntegerField(default=0)),
                ('infracciones_luz_roja', models.IntegerField(default=0)),
                ('infracciones_velocidad', models.IntegerField(default=0)),
                ('infracciones_graves', models.IntegerField(default=0)),
                ('puntuacion_riesgo', models.DecimalField(decimal_places=2, default=0.0, help_text='0-100', max_digits=5)),
                ('nivel_riesgo', models.CharField(choices=[('BAJO', 'Bajo'), ('MEDIO', 'Medio'), ('ALTO', 'Alto'), ('CRITICO', 'Crítico')], default='BAJO', max_length=20)),
                ('probabilidad_reincidencia', models.DecimalField(decimal_places=2, default=0.0, help_text='Porcentaje', max_digits=5)),
                ('probabilidad_accidente', models.DecimalField(decimal_places=2, default=0.0, help_text='Porcentaje', max_digits=5)),
                ('ultima_actualizacion', models.DateTimeField(auto_now=True)),
                ('vehiculo', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='perfil', to='infracciones.vehiculo')),
            ],
            options={
                'verbose_name': 'Perfil de Conductor',
                'verbose_name_plural': 'Perfiles de Conductores',
            },
        ),
        migrations.CreateModel(
            name='PrediccionAccidente',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ubicacion', models.CharField(max_length=300)),
                ('latitud', models.DecimalField(decimal_places=6, max_digits=9)),
                ('longitud', models.DecimalField(decimal_places=6, max_digits=9)),
                ('fecha_prediccion', models.DateTimeField(auto_now_add=True)),
                ('periodo_prediccion', models.CharField(choices=[('PROXIMA_HORA', 'Próxima Hora'), ('PROXIMO_DIA', 'Próximo Día'), ('PROXIMA_SEMANA', 'Próxima Semana'), ('PROXIMO_MES', 'Próximo Mes')], max_length=50)),
                ('probabilidad', models.DecimalField(decimal_places=2, help_text='Porcentaje', max_digits=5)),
                ('factores_riesgo', models.JSONField(help_text='Factores que contribuyen al riesgo')),
                ('infracciones_historicas', models.IntegerField(default=0)),
                ('accidentes_historicos', models.IntegerField(default=0)),
                ('modelo_version', models.CharField(default='v1.0', max_length=50)),
            ],
            options={
                'verbose_name': 'Predicción de Accidente',
                'verbose_name_plural': 'Predicciones de Accidentes',
                'ordering': ['-probabilidad', '-fecha_prediccion'],
            },
        ),
        migrations.CreateModel(
            name='EventoDeteccion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True, db_index=True)),
                ('tipo_evento', models.CharField(choices=[('VEHICULO_DETECTADO', 'Vehículo Detectado'), ('PLACA_RECONOCIDA', 'Placa Reconocida'), ('INFRACCION_DETECTADA', 'Infracción Detectada'), ('ERROR_DETECCION', 'Error de Detección'), ('CAMARA_OFFLINE', 'Cámara Offline')], max_length=50)),
                ('datos_evento', models.JSONField(help_text='Datos adicionales del evento')),
                ('imagen_frame', models.ImageField(blank=True, null=True, upload_to='eventos/frames/')),
                ('camara', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='eventos', to='camaras.camara')),
            ],
            options={
                'verbose_name': 'Evento de Detección',
                'verbose_name_plural': 'Eventos de Detección',
                'ordering': ['-timestamp'],
            },
        ),
        migrations.AddIndex(
            model_name='infraccion',
            index=models.Index(fields=['fecha_hora', 'estado'], name='infraccione_fecha_h_8e5c8a_idx'),
        ),
        migrations.AddIndex(
            model_name='infraccion',
            index=models.Index(fields=['vehiculo', 'fecha_hora'], name='infraccione_vehicul_9a7b2c_idx'),
        ),
    ]

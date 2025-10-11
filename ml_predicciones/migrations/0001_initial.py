from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ModeloEntrenamiento',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nombre', models.CharField(max_length=200)),
                ('version', models.CharField(max_length=50)),
                ('tipo_modelo', models.CharField(choices=[('CLASIFICACION', 'Clasificación'), ('REGRESION', 'Regresión'), ('CLUSTERING', 'Clustering'), ('DETECCION_OBJETOS', 'Detección de Objetos'), ('RECONOCIMIENTO_PLACAS', 'Reconocimiento de Placas')], max_length=50)),
                ('objetivo', models.TextField(help_text='Qué predice este modelo')),
                ('accuracy', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('precision', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('recall', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('f1_score', models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ('archivo_modelo', models.FileField(blank=True, null=True, upload_to='modelos/')),
                ('archivo_pesos', models.FileField(blank=True, null=True, upload_to='modelos/pesos/')),
                ('fecha_entrenamiento', models.DateTimeField(auto_now_add=True)),
                ('dataset_size', models.IntegerField(help_text='Cantidad de datos de entrenamiento')),
                ('activo', models.BooleanField(default=False)),
                ('notas', models.TextField(blank=True, null=True)),
            ],
            options={
                'verbose_name': 'Modelo de Entrenamiento',
                'verbose_name_plural': 'Modelos de Entrenamiento',
                'ordering': ['-fecha_entrenamiento'],
            },
        ),
        migrations.CreateModel(
            name='DatasetEntrenamiento',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nombre', models.CharField(max_length=200)),
                ('descripcion', models.TextField()),
                ('tipo_datos', models.CharField(choices=[('IMAGENES', 'Imágenes'), ('VIDEOS', 'Videos'), ('INFRACCIONES', 'Datos de Infracciones'), ('MIXTO', 'Mixto')], max_length=50)),
                ('cantidad_registros', models.IntegerField()),
                ('fecha_creacion', models.DateTimeField(auto_now_add=True)),
                ('archivo_dataset', models.FileField(blank=True, null=True, upload_to='datasets/')),
                ('etiquetado_completo', models.BooleanField(default=False)),
            ],
            options={
                'verbose_name': 'Dataset de Entrenamiento',
                'verbose_name_plural': 'Datasets de Entrenamiento',
            },
        ),
    ]

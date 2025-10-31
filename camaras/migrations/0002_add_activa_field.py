from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('camaras', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='camara',
            name='activa',
            field=models.BooleanField(default=True, help_text='Indica si la cámara está activa y operativa'),
        ),
        migrations.AddField(
            model_name='camara',
            name='fecha_instalacion',
            field=models.DateTimeField(auto_now_add=True, null=True, blank=True),
        ),
        migrations.AddField(
            model_name='camara',
            name='ultima_conexion',
            field=models.DateTimeField(null=True, blank=True),
        ),
    ]

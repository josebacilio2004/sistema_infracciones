from django import forms
from .models import Camara

class CamaraForm(forms.ModelForm):
    class Meta:
        model = Camara
        fields = ["ubicacion", "descripcion"]  # Ajusta según los campos de tu modelo

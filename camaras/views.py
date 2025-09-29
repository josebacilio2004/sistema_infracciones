from django.shortcuts import render, get_object_or_404, redirect
from .models import Camara
from .forms import CamaraForm

def lista_camaras(request):
    camaras = Camara.objects.all()
    return render(request, "camaras/lista.html", {"camaras": camaras})

def crear_camara(request):
    if request.method == "POST":
        form = CamaraForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("camaras_lista")
    else:
        form = CamaraForm()
    return render(request, "camaras/form.html", {"form": form, "titulo": "Crear Cámara"})

def editar_camara(request, pk):
    camara = get_object_or_404(Camara, pk=pk)
    if request.method == "POST":
        form = CamaraForm(request.POST, instance=camara)
        if form.is_valid():
            form.save()
            return redirect("camaras_lista")
    else:
        form = CamaraForm(instance=camara)
    return render(request, "camaras/form.html", {"form": form, "titulo": "Editar Cámara"})

def eliminar_camara(request, pk):
    camara = get_object_or_404(Camara, pk=pk)
    if request.method == "POST":
        camara.delete()
        return redirect("camaras_lista")
    return render(request, "camaras/confirmar_eliminar.html", {"camara": camara})

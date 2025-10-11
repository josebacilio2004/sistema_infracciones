#!/bin/bash

# Script para aplicar las migraciones de la base de datos

echo "Aplicando migraciones..."
python manage.py makemigrations
python manage.py migrate

echo "Migraciones aplicadas exitosamente!"

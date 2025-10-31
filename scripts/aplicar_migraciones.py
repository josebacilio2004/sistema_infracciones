#!/usr/bin/env python
"""
Script para aplicar migraciones de Django
"""
import os
import sys
import django

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'seguridad.settings')
django.setup()

from django.core.management import call_command

def main():
    print("=" * 50)
    print("Aplicando migraciones de la base de datos")
    print("=" * 50)
    
    try:
        # Crear migraciones
        print("\n1. Creando migraciones...")
        call_command('makemigrations')
        
        # Aplicar migraciones
        print("\n2. Aplicando migraciones...")
        call_command('migrate')
        
        print("\n" + "=" * 50)
        print("✓ Migraciones aplicadas exitosamente!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error al aplicar migraciones: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

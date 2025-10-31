"""
Utilidades para procesamiento de placas vehiculares peruanas
"""
import re
from typing import Optional, Tuple

# Formato de placa peruana
PATRON_PLACA_PERU = re.compile(r'^[A-Z0-9]{3}-[0-9]{3}$')

# Caracteres comúnmente confundidos en OCR
CORRECCIONES_OCR = {
    'O': '0',  # O → 0
    'I': '1',  # I → 1
    'Z': '2',  # Z → 2
    'S': '5',  # S → 5
    'B': '8',  # B → 8
}

def limpiar_texto_placa(texto: str) -> str:
    """
    Limpia el texto extraído por OCR
    
    Args:
        texto: Texto crudo del OCR
        
    Returns:
        Texto limpio en mayúsculas sin espacios
    """
    # Convertir a mayúsculas
    texto = texto.upper().strip()
    
    # Remover espacios
    texto = texto.replace(' ', '')
    
    # Remover caracteres especiales excepto guión
    texto = re.sub(r'[^A-Z0-9-]', '', texto)
    
    return texto

def validar_placa_peruana(placa: str) -> bool:
    """
    Valida si una placa cumple con el formato peruano
    
    Formato: A1B-234
    - 3 caracteres alfanuméricos
    - Guión
    - 3 números
    
    Args:
        placa: Texto de la placa
        
    Returns:
        True si es válida, False en caso contrario
    """
    placa = limpiar_texto_placa(placa)
    
    # Validar con patrón
    if PATRON_PLACA_PERU.match(placa):
        return True
    
    # Intentar con formato sin guión
    if len(placa) == 6:
        parte1 = placa[:3]
        parte2 = placa[3:]
        if parte1.isalnum() and parte2.isdigit():
            return True
    
    return False

def normalizar_placa_peruana(texto: str) -> Optional[str]:
    """
    Normaliza el texto de una placa al formato estándar peruano
    
    Args:
        texto: Texto extraído por OCR
        
    Returns:
        Placa normalizada (A1B-234) o None si no es válida
    """
    texto = limpiar_texto_placa(texto)
    
    # Si ya tiene el formato correcto
    if PATRON_PLACA_PERU.match(texto):
        return texto
    
    # Intentar agregar guión
    if len(texto) == 6:
        parte1 = texto[:3]
        parte2 = texto[3:]
        
        if parte1.isalnum() and parte2.isdigit():
            return f"{parte1}-{parte2}"
    
    # Intentar corregir errores comunes de OCR en la parte numérica
    if len(texto) >= 6:
        # Separar por guión si existe
        if '-' in texto:
            partes = texto.split('-')
            if len(partes) == 2:
                parte1, parte2 = partes
                
                # Corregir parte numérica
                parte2_corregida = ''
                for char in parte2[:3]:
                    if char in CORRECCIONES_OCR and len(parte2_corregida) >= 0:
                        parte2_corregida += CORRECCIONES_OCR[char]
                    elif char.isdigit():
                        parte2_corregida += char
                
                if len(parte1) == 3 and len(parte2_corregida) == 3:
                    return f"{parte1}-{parte2_corregida}"
    
    return None

def extraer_info_placa(placa: str) -> Optional[dict]:
    """
    Extrae información de una placa peruana
    
    Args:
        placa: Placa normalizada (A1B-234)
        
    Returns:
        Diccionario con información o None
    """
    placa = normalizar_placa_peruana(placa)
    
    if not placa:
        return None
    
    partes = placa.split('-')
    if len(partes) != 2:
        return None
    
    codigo, numero = partes
    
    return {
        'placa_completa': placa,
        'codigo': codigo,
        'numero': numero,
        'formato': 'PERUANO',
        'valida': True
    }

def generar_placa_ejemplo() -> str:
    """
    Genera una placa de ejemplo para pruebas
    
    Returns:
        Placa de ejemplo en formato peruano
    """
    import random
    import string
    
    # Generar código alfanumérico
    codigo = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    
    # Generar número
    numero = ''.join(random.choices(string.digits, k=3))
    
    return f"{codigo}-{numero}"

def es_placa_similar(placa1: str, placa2: str, tolerancia: int = 1) -> bool:
    """
    Compara dos placas y determina si son similares
    Útil para tracking de vehículos con OCR imperfecto
    
    Args:
        placa1: Primera placa
        placa2: Segunda placa
        tolerancia: Número máximo de caracteres diferentes permitidos
        
    Returns:
        True si son similares, False en caso contrario
    """
    placa1 = limpiar_texto_placa(placa1)
    placa2 = limpiar_texto_placa(placa2)
    
    if len(placa1) != len(placa2):
        return False
    
    diferencias = sum(c1 != c2 for c1, c2 in zip(placa1, placa2))
    
    return diferencias <= tolerancia


# Ejemplos de uso
if __name__ == "__main__":
    print("🇵🇪 Utilidades para Placas Peruanas\n")
    
    # Ejemplos de validación
    placas_prueba = [
        "AEF-717",
        "AEF717",
        "B2C-456",
        "XYZ123",
        "ABC-12",  # Inválida
        "ABCD-123",  # Inválida
    ]
    
    print("Validación de placas:")
    for placa in placas_prueba:
        valida = validar_placa_peruana(placa)
        normalizada = normalizar_placa_peruana(placa)
        print(f"  {placa:12} → Válida: {valida:5} | Normalizada: {normalizada}")
    
    print("\nGeneración de placa de ejemplo:")
    for _ in range(5):
        print(f"  {generar_placa_ejemplo()}")
    
    print("\nComparación de similitud:")
    pares = [
        ("AEF-717", "AEF-717"),
        ("AEF-717", "AEF-71O"),  # O en lugar de 0
        ("AEF-717", "AEF-718"),
        ("AEF-717", "BEF-717"),
    ]
    
    for p1, p2 in pares:
        similar = es_placa_similar(p1, p2)
        print(f"  {p1} vs {p2}: {'Similar' if similar else 'Diferente'}")

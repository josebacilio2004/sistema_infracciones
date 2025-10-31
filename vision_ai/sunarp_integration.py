"""
INTEGRACIÓN CON SUNARP (Simulada) - PARA DESARROLLO
Versión mejorada con base de datos simulada de vehículos peruanos
"""

import logging
import random
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class SunarpConsultor:
    """Consultor SIMULADO de datos vehiculares - Para desarrollo"""
    
    def __init__(self):
        # Base de datos simulada extensa de vehículos peruanos
        self.vehiculos_mock = {
            # Formatos tradicionales ABC-123
            'ABC123': {
                'placa': 'ABC-123',
                'marca': 'TOYOTA',
                'modelo': 'COROLLA',
                'color': 'BLANCO',
                'anio': 2022,
                'propietario_nombre': 'JUAN PEREZ RODRIGUEZ',
                'numero_motor': '2ZR1234567',
                'numero_serie': 'JTDBR123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'AUTO',
                'fuente': 'SUNARP_SIMULADO'
            },
            'XYZ789': {
                'placa': 'XYZ-789',
                'marca': 'NISSAN',
                'modelo': 'SENTRA', 
                'color': 'NEGRO',
                'anio': 2021,
                'propietario_nombre': 'MARIA GARCIA LOPEZ',
                'numero_motor': 'HR161234567',
                'numero_serie': '3N1BC123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'AUTO',
                'fuente': 'SUNARP_SIMULADO'
            },
            # Nuevos formatos X71-962
            'X71962': {
                'placa': 'X71-962',
                'marca': 'HYUNDAI',
                'modelo': 'TUCSON',
                'color': 'GRIS',
                'anio': 2023,
                'propietario_nombre': 'CARLOS LOPEZ MARTINEZ',
                'numero_motor': 'G4NJ1234567',
                'numero_serie': 'KM8J3123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'CAMIONETA',
                'fuente': 'SUNARP_SIMULADO'
            },
            'Y82471': {
                'placa': 'Y82-471',
                'marca': 'MITSUBISHI',
                'modelo': 'OUTLANDER',
                'color': 'BLANCO',
                'anio': 2021,
                'propietario_nombre': 'JORGE TAPIA FLORES',
                'numero_motor': '4B121234567',
                'numero_serie': 'JA4MW123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'CAMIONETA',
                'fuente': 'SUNARP_SIMULADO'
            },
            'Z93582': {
                'placa': 'Z93-582',
                'marca': 'SUBARU',
                'modelo': 'FORESTER',
                'color': 'AZUL',
                'anio': 2022,
                'propietario_nombre': 'ELENA MORALES CASTRO',
                'numero_motor': 'FB201234567',
                'numero_serie': 'JF2SJ123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'CAMIONETA',
                'fuente': 'SUNARP_SIMULADO'
            },
            # Más vehículos comunes en Perú
            'AB1234': {
                'placa': 'AB-1234',
                'marca': 'KIA',
                'modelo': 'RIO',
                'color': 'ROJO',
                'anio': 2020,
                'propietario_nombre': 'ANA TORRES SANCHEZ',
                'numero_motor': 'G4F1234567',
                'numero_serie': 'KNAFX123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'AUTO',
                'fuente': 'SUNARP_SIMULADO'
            },
            'CD5678': {
                'placa': 'CD-5678',
                'marca': 'SUZUKI',
                'modelo': 'SWIFT',
                'color': 'AZUL',
                'anio': 2019,
                'propietario_nombre': 'PEDRO RAMIREZ CASTILLO',
                'numero_motor': 'K12B1234567',
                'numero_serie': 'MS0BX123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'AUTO',
                'fuente': 'SUNARP_SIMULADO'
            },
            'EF9012': {
                'placa': 'EF-9012',
                'marca': 'HONDA',
                'modelo': 'CIVIC',
                'color': 'PLATEADO',
                'anio': 2021,
                'propietario_nombre': 'LUISA FERNANDEZ GOMEZ',
                'numero_motor': 'R18A1234567',
                'numero_serie': '2HGFG123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'AUTO',
                'fuente': 'SUNARP_SIMULADO'
            },
            # Motocicletas
            'M12345': {
                'placa': 'M-12345',
                'marca': 'YAMAHA',
                'modelo': 'YZF-R3',
                'color': 'AZUL',
                'anio': 2022,
                'propietario_nombre': 'ROBERTO SILVA MENDOZA',
                'numero_motor': 'R31234567',
                'numero_serie': 'JYARN123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'MOTO',
                'fuente': 'SUNARP_SIMULADO'
            },
            'M67890': {
                'placa': 'M-67890',
                'marca': 'KAWASAKI',
                'modelo': 'NINJA 400',
                'color': 'VERDE',
                'anio': 2023,
                'propietario_nombre': 'DIEGO CASTRO RAMOS',
                'numero_motor': 'EX4001234567',
                'numero_serie': 'JKAEX123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'MOTO',
                'fuente': 'SUNARP_SIMULADO'
            },
            # Vehículos comerciales
            'T11223': {
                'placa': 'T-11223',
                'marca': 'HINO',
                'modelo': '300 SERIES',
                'color': 'BLANCO',
                'anio': 2020,
                'propietario_nombre': 'TRANSPORTES ANDINOS SAC',
                'numero_motor': 'NO4C1234567',
                'numero_serie': 'JHDFB123456789012',
                'estado': 'ACTIVO',
                'tipo_vehiculo': 'CAMION',
                'fuente': 'SUNARP_SIMULADO'
            }
        }
    
    @staticmethod
    def normalizar_placa(placa: str) -> str:
        """Normaliza la placa (elimina guiones y espacios)"""
        return placa.upper().replace('-', '').replace(' ', '').strip()
    
    @staticmethod
    def validar_placa_peruana(placa: str) -> bool:
        """Valida formatos de placa peruana"""
        placa_limpia = SunarpConsultor.normalizar_placa(placa)
        
        patrones = [
            r'^[A-Z]{3}\d{3}$',      # ABC123
            r'^[A-Z]{2}\d{3}$',      # AB123
            r'^[A-Z]{2}\d{4}$',      # AB1234
            r'^[A-Z]\d{5}$',         # A12345 (nuevo formato)
            r'^[A-Z]\d{2}\d{3}$',    # X71962
            r'^M\d{5}$',             # M12345 (motos)
            r'^T\d{5}$',             # T12345 (taxi/comercial)
        ]
        
        return any(re.match(patron, placa_limpia) for patron in patrones)
    
    def consultar(self, placa: str) -> Optional[Dict]:
        """
        Consulta SIMULADA a SUNARP - Para desarrollo
        
        Args:
            placa: Placa del vehículo (ej: ABC-123, X71-962, etc.)
        
        Returns:
            Dict con datos del vehículo o None si no se encuentra
        """
        try:
            placa_normalizada = self.normalizar_placa(placa)
            
            if not self.validar_placa_peruana(placa_normalizada):
                logger.warning(f"Placa inválida: {placa} -> {placa_normalizada}")
                return None
            
            logger.info(f"🔍 Consultando SUNARP (SIMULADO) para: {placa_normalizada}")
            
            # Simular delay de red
            import time
            time.sleep(0.5)
            
            # Buscar en base de datos simulada
            if placa_normalizada in self.vehiculos_mock:
                datos = self.vehiculos_mock[placa_normalizada].copy()
                datos['fecha_consulta'] = datetime.now().isoformat()
                datos['estado_consulta'] = 'EXITOSA'
                
                logger.info(f"✅ Vehículo encontrado: {datos['placa']} - {datos['marca']} {datos['modelo']}")
                return datos
            
            # Si no está en la base simulada, generar datos aleatorios
            logger.warning(f"⚠️ Vehículo no encontrado, generando datos simulados para: {placa_normalizada}")
            return self._generar_datos_simulados(placa_normalizada)
            
        except Exception as e:
            logger.error(f"❌ Error en consulta SUNARP simulada: {str(e)}")
            return None
    
    def _generar_datos_simulados(self, placa: str) -> Dict:
        """Genera datos simulados para placas no encontradas"""
        marcas = ['TOYOTA', 'NISSAN', 'HYUNDAI', 'KIA', 'SUZUKI', 'HONDA', 'MAZDA']
        modelos = {
            'TOYOTA': ['COROLLA', 'HILUX', 'RAV4', 'YARIS'],
            'NISSAN': ['SENTRA', 'VERSA', 'X-TRAIL', 'FRONTIER'],
            'HYUNDAI': ['TUCSON', 'CRETA', 'ACCENT', 'ELANTRA'],
            'KIA': ['RIO', 'PICANTO', 'SELTOS', 'SPORTAGE'],
            'SUZUKI': ['SWIFT', 'VITARA', 'S-CROSS', 'JIMNY'],
            'HONDA': ['CIVIC', 'CR-V', 'HR-V', 'ACCORD'],
            'MAZDA': ['3', 'CX-5', 'CX-30', '2']
        }
        colores = ['BLANCO', 'NEGRO', 'GRIS', 'PLATEADO', 'AZUL', 'ROJO', 'VERDE']
        
        marca = random.choice(marcas)
        modelo = random.choice(modelos[marca])
        color = random.choice(colores)
        anio = random.randint(2018, 2024)
        
        return {
            'placa': placa,
            'marca': marca,
            'modelo': modelo,
            'color': color,
            'anio': anio,
            'propietario_nombre': 'PROPIETARIO SIMULADO',
            'numero_motor': f'{marca[:2]}{random.randint(1000000, 9999999)}',
            'numero_serie': f'SIMULATED{random.randint(100000000000, 999999999999)}',
            'estado': 'ACTIVO',
            'tipo_vehiculo': 'AUTO',
            'fuente': 'SUNARP_SIMULADO',
            'fecha_consulta': datetime.now().isoformat(),
            'estado_consulta': 'SIMULADO',
            'nota': 'Datos generados automáticamente para desarrollo'
        }


class GeneradorInfraccion:
    """Genera infracciones automáticamente basado en detecciones"""
    
    TIPOS_INFRACCION = {
        'EXCESO_VELOCIDAD': {
            'codigo': 'EV001',
            'nombre': 'Exceso de Velocidad',
            'multa': 1500.00,
            'puntos': 15
        },
        'LUZ_ROJA': {
            'codigo': 'LR001',
            'nombre': 'Pasarse Luz Roja',
            'multa': 2000.00,
            'puntos': 20
        },
        'INVASION_CARRIL': {
            'codigo': 'IC001',
            'nombre': 'Invasión de Carril Contrario',
            'multa': 1800.00,
            'puntos': 18
        }
    }
    
    @staticmethod
    def crear_infraccion(vehiculo, tipo_infraccion, ubicacion, 
                        velocidad_detectada=None, confianza=0.95):
        """
        Crea un registro de infracción en la BD
        
        Args:
            vehiculo: Objeto Vehiculo
            tipo_infraccion: Tipo de infracción (EXCESO_VELOCIDAD, LUZ_ROJA, INVASION_CARRIL)
            ubicacion: Ubicación donde se detectó
            velocidad_detectada: Velocidad en km/h (opcional)
            confianza: Confianza del modelo (0-1)
        
        Returns:
            Objeto Infraccion creado
        """
        from infracciones.models import Infraccion, TipoInfraccion
        from django.utils import timezone
        
        try:
            # Obtener o crear tipo de infracción
            tipo_info = GeneradorInfraccion.TIPOS_INFRACCION.get(tipo_infraccion)
            
            if not tipo_info:
                logger.error(f"Tipo de infracción desconocido: {tipo_infraccion}")
                return None
            
            tipo_obj, _ = TipoInfraccion.objects.get_or_create(
                codigo=tipo_info['codigo'],
                defaults={
                    'nombre': tipo_info['nombre'],
                    'descripcion': f'Infracción detectada automáticamente: {tipo_info["nombre"]}',
                    'monto_multa': tipo_info['multa'],
                    'puntos_licencia': tipo_info['puntos'],
                    'gravedad': 'GRAVE' if tipo_info['puntos'] >= 18 else 'MODERADA'
                }
            )
            
            # Crear infracción
            infraccion = Infraccion.objects.create(
                vehiculo=vehiculo,
                tipo_infraccion=tipo_obj,
                fecha_hora=timezone.now(),
                ubicacion=ubicacion,
                velocidad_detectada=velocidad_detectada,
                confianza_deteccion=confianza * 100,
                estado='DETECTADA',
                modelo_ia_version='v2.0'
            )
            
            logger.info(f"✅ Infracción creada: {infraccion.id} para placa {vehiculo.placa}")
            return infraccion
        
        except Exception as e:
            logger.error(f"❌ Error creando infracción: {str(e)}")
            return None

# Importar regex para validaciones
import re
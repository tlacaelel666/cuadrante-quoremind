# __init__.py en el directorio de configuraciones

import os
import sys
import logging
from typing import Dict, Any, Optional
from functools import lru_cache

import yaml
import dotenv

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('config_manager.log')
    ]
)
logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Gestor centralizado de configuraciones para proyectos de ML y Quantum Computing.
    
    Características principales:
    - Carga de configuraciones desde YAML
    - Soporte para variables de entorno
    - Caché de configuraciones
    - Validación de configuraciones
    - Gestión de secretos
    """
    
    _instance = None
    _config_cache = {}
    
    def __new__(cls):
        """Implementación de Singleton para asegurar una única instancia."""
        if not cls._instance:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Inicializa el gestor de configuraciones.
        
        Args:
            config_dir: Directorio de configuraciones. 
                        Si es None, usa el directorio del script actual.
        """
        if not hasattr(self, 'initialized'):
            self.config_dir = config_dir or os.path.dirname(os.path.abspath(__file__))
            self.env_file = os.path.join(self.config_dir, '..', '.env')
            
            # Cargar variables de entorno
            self._load_env_variables()
            
            # Inicializar caché
            self._config_cache = {}
            
            self.initialized = True
    
    def _load_env_variables(self):
        """
        Carga variables de entorno desde un archivo .env.
        Soporta múltiples archivos de entorno según el entorno.
        """
        try:
            # Cargar .env base
            if os.path.exists(self.env_file):
                dotenv.load_dotenv(self.env_file)
            
            # Cargar .env específico del entorno
            env = os.getenv('ENVIRONMENT', 'development')
            env_specific_file = os.path.join(
                self.config_dir, 
                f'.env.{env}'
            )
            
            if os.path.exists(env_specific_file):
                dotenv.load_dotenv(env_specific_file, override=True)
        except Exception as e:
            logger.error(f"Error cargando variables de entorno: {e}")
    
    @lru_cache(maxsize=32)
    def load_config(
        self, 
        config_name: str, 
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Carga una configuración específica con opciones de validación.
        
        Args:
            config_name: Nombre del archivo de configuración (ej. 'qiskit')
            validate: Si se debe validar la configuración
        
        Returns:
            Diccionario de configuración
        """
        try:
            # Buscar archivo de configuración
            config_file = os.path.join(
                self.config_dir, 
                f'{config_name}_config.yaml'
            )
            
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_file}")
            
            # Cargar configuración
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            
            # Reemplazar variables de entorno
            config = self._replace_env_vars(config)
            
            # Validar configuración si está habilitado
            if validate:
                config = self._validate_config(config_name, config)
            
            return config
        except Exception as e:
            logger.error(f"Error cargando configuración {config_name}: {e}")
            return {}
    
    def _replace_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reemplaza marcadores de variables de entorno en la configuración.
        
        Ejemplo:
        qiskit:
          token: ${QISKIT_TOKEN}
        """
        def _replace_recursive(value):
            if isinstance(value, dict):
                return {k: _replace_recursive(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_replace_recursive(v) for v in value]
            elif isinstance(value, str):
                # Reemplazar variables de entorno
                if value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    return os.getenv(env_var, value)
            return value
        
        return _replace_recursive(config)
    
    def _validate_config(
        self, 
        config_name: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Valida la configuración según reglas específicas.
        
        Args:
            config_name: Nombre de la configuración
            config: Diccionario de configuración
        
        Returns:
            Configuración validada
        """
        validation_rules = {
            'qiskit': [
                ('credentials.token', str, None),
                ('runtime.shots', int, 1024),
                ('runtime.max_execution_time', int, 3600)
            ],
            'tensorflow': [
                ('gpu.memory_limit', int, 4096),
                ('training.learning_rate', (int, float), 0.001)
            ],
            'pytorch': [
                ('device.default', str, 'cuda'),
                ('training.learning_rate', (int, float), 0.001)
            ]
        }
        
        # Obtener reglas de validación
        rules = validation_rules.get(config_name, [])
        
        for path, expected_type, default in rules:
            try:
                # Navegar por la configuración anidada
                current = config
                parts = path.split('.')
                for part in parts[:-1]:
                    current = current.get(part, {})
                
                # Obtener valor
                value = current.get(parts[-1])
                
                # Validar tipo
                if value is None:
                    # Usar valor por defecto si no está definido
                    self._set_nested_value(config, path, default)
                elif not isinstance(value, expected_type):
                    logger.warning(
                        f"Valor inválido para {path}. "
                        f"Esperado {expected_type}, encontrado {type(value)}"
                    )
                    # Convertir o usar valor por defecto
                    self._set_nested_value(
                        config, 
                        path, 
                        default if default is not None else value
                    )
            except Exception as e:
                logger.error(f"Error validando {path}: {e}")
        
        return config
    
    def _set_nested_value(
        self, 
        config: Dict[str, Any], 
        path: str, 
        value: Any
    ):
        """
        Establece un valor en una configuración anidada.
        
        Args:
            config: Diccionario de configuración
            path: Ruta al valor (ej. 'qiskit.credentials.token')
            value: Valor a establecer
        """
        parts = path.split('.')
        current = config
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Obtiene un secreto de variables de entorno.
        
        Args:
            secret_name: Nombre del secreto
        
        Returns:
            Valor del secreto o None
        """
        return os.getenv(secret_name)
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusiona múltiples configuraciones.
        
        Args:
            *configs: Configuraciones a fusionar
        
        Returns:
            Configuración fusionada
        """
        merged = {}
        for config in configs:
            self._deep_update(merged, config)
        return merged
    
    def _deep_update(
        self, 
        base: Dict[str, Any], 
        update: Dict[str, Any]
    ):
        """
        Actualización profunda de diccionarios.
        
        Args:
            base: Diccionario base
            update: Diccionario con actualizaciones
        """
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = self._deep_update(
                    base.get(key, {}), 
                    value
                )
            else:
                base[key] = value
        return base

# Instancia global
config_manager = ConfigurationManager()

# Funciones de conveniencia
def load_config(config_name: str, validate: bool = True) -> Dict[str, Any]:
    """
    Función de conveniencia para cargar configuración.
    
    Args:
        config_name: Nombre de la configuración
        validate: Si se debe validar la configuración
    
    Returns:
        Configuración cargada
    """
    return config_manager.load_config(config_name, validate)

def get_secret(secret_name: str) -> Optional[str]:
    """
    Función de conveniencia para obtener secretos.
    
    Args:
        secret_name: Nombre del secreto
    
    Returns:
        Valor del secreto
    """
    return config_manager.get_secret(secret_name)

# Ejemplo de uso
def main():
    # Cargar configuraciones
    qiskit_config = load_config('qiskit')
    tensorflow_config = load_config('tensorflow')
    
    # Obtener un secreto
    qiskit_token = get_secret('QISKIT_TOKEN')
    
    # Fusionar configuraciones
    merged_config = config_manager.merge_configs(
        qiskit_config, 
        tensorflow_config
    )
    
    # Imprimir configuración
    import json
    print(json.dumps(merged_config, indent=2))

if __name__ == '__main__':
    main()
import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    """Clase para cargar y gestionar configuraciones."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Carga un archivo de configuración YAML.
        
        Args:
            config_path (str): Ruta al archivo de configuración.
        
        Returns:
            Dict con la configuración cargada.
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Archivo de configuración no encontrado: {config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error al parsear el archivo YAML: {e}")
            return {}

    @staticmethod
    def get_config_path(filename: str) -> str:
        """
        Obtiene la ruta completa de un archivo de configuración.
        
        Args:
            filename (str): Nombre del archivo de configuración.
        
        Returns:
            Ruta completa al archivo de configuración.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, 'configs', filename)

    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusiona múltiples configuraciones.
        
        Args:
            *configs: Diccionarios de configuración a fusionar.
        
        Returns:
            Diccionario de configuración fusionado.
        """
        merged_config = {}
        for config in configs:
            for key, value in config.items():
                if isinstance(value, dict):
                    merged_config[key] = merged_config.get(key, {})
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
        return merged_config

# Ejemplo de uso
def main():
    # Cargar configuraciones
    qiskit_config = ConfigLoader.load_config(
        ConfigLoader.get_config_path('qiskit_config.yaml')
    )
    tensorflow_config = ConfigLoader.load_config(
        ConfigLoader.get_config_path('tensorflow_config.yaml')
    )
    
    # Fusionar configuraciones
    merged_config = ConfigLoader.merge_configs(
        qiskit_config, 
        tensorflow_config
    )
    
    # Imprimir configuración fusionada
    import json
    print(json.dumps(merged_config, indent=2))

if __name__ == '__main__':
    main()
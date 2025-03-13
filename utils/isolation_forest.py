import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

class Honeypot:
    def __init__(self, name, threshold=0.5):
        self.name = name
        self.threshold = threshold
        self.interactions = []

    def record_interaction(self, interaction):
        """Registra interacciones capturadas por el honeypot"""
        self.interactions.append(interaction)
        print(f"[{self.name}] Interacción registrada: {interaction}")

    def export_logs(self):
        """Exporta los logs de interacciones"""
        return self.interactions


class SecuritySystem:
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        self.honeypots = {}  # Honeypots para atrapar actividades sospechosas
        self.isolation_forest = IsolationForest(
            n_estimators=n_estimators, 
            contamination=contamination, 
            random_state=random_state
        )
        self.interactions_log = []  # Registro de interacciones para entrenamiento
        self.model_trained = False

    def add_honeypot(self, name, threshold=0.5):
        """Agrega un honeypot para monitorear actividades sospechosas."""
        self.honeypots[name] = Honeypot(name, threshold)
        return self.honeypots[name]

    @staticmethod
    def sanitize_input(interaction):
        """Sanitiza la entrada eliminando contenido malicioso o inválido."""
        if not isinstance(interaction, dict):
            return {}
        sanitized_interaction = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in interaction.items()}
        return sanitized_interaction

    def _prepare_feature_vector(self, interaction):
        """Convierte la interacción en un vector numérico para el modelo."""
        if not interaction:
            return []
        # Usar un método más robusto para vectorizar características
        return [hash(f"{key}:{interaction[key]}") % 1e6 for key in sorted(interaction.keys())]

    def detect_anomalies(self, interaction):
        """Detecta anomalías usando Isolation Forest."""
        if not self.model_trained or not self.interactions_log:
            return False  # Sin modelo entrenado o datos previos, no se detectan anomalías.

        # Convertir la interacción a un vector numérico
        interaction_vector = self._prepare_feature_vector(interaction)
        if not interaction_vector:
            return False
        
        try:
            # -1 indica anomalía, 1 indica comportamiento normal
            return self.isolation_forest.predict([interaction_vector])[0] == -1
        except Exception as e:
            print(f"Error al detectar anomalías: {e}")
            return False

    @staticmethod
    def block_user(user_id):
        """Bloquea a un usuario identificado como malicioso."""
        print(f"Usuario {user_id} bloqueado por comportamiento sospechoso.")
        # Implementar la lógica de bloqueo (e.g., actualizar base de datos, marcar sesión, etc.)

    def train_model(self):
        """Entrena el modelo con las interacciones registradas."""
        if len(self.interactions_log) < 10:
            print("No hay suficientes datos para entrenar el modelo (mínimo 10 requeridos).")
            return False

        try:
            # Crear un conjunto de datos basado en el log de interacciones
            data = [self._prepare_feature_vector(interaction) for interaction in self.interactions_log]
            # Filtrar vectores vacíos
            data = [vector for vector in data if vector]
            
            if not data:
                print("No se pudieron extraer características válidas de las interacciones.")
                return False
                
            # Asegurar que todos los vectores tienen la misma longitud
            max_length = max(len(vector) for vector in data)
            padded_data = [vector + [0] * (max_length - len(vector)) for vector in data]
            
            self.isolation_forest.fit(padded_data)
            self.model_trained = True
            print(f"Modelo de seguridad entrenado con éxito usando {len(data)} interacciones.")
            return True
        except Exception as e:
            print(f"Error al entrenar el modelo: {e}")
            return False

    def log_interaction(self, interaction):
        """Registra una interacción para monitoreo y entrenamiento futuro."""
        sanitized_interaction = self.sanitize_input(interaction)
        if sanitized_interaction:
            self.interactions_log.append(sanitized_interaction)

    def handle_security(self, interaction, user_id):
        """Sanitiza, registra y verifica la interacción."""
        sanitized_interaction = self.sanitize_input(interaction)
        self.log_interaction(sanitized_interaction)

        # Registrar en honeypots si existen
        for honeypot in self.honeypots.values():
            honeypot.record_interaction(sanitized_interaction)

        if self.detect_anomalies(sanitized_interaction):
            self.block_user(user_id)
            return False  # Indica que la interacción fue bloqueada

        return True  # Indica que la interacción es segura

    def save_model(self, path="isolation_forest.pkl"):
        """Guarda el modelo entrenado en un archivo."""
        if not self.model_trained:
            print("No hay modelo entrenado para guardar.")
            return False
            
        try:
            joblib.dump(self.isolation_forest, path)
            print(f"Modelo guardado en {path}")
            return True
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
            return False

    def load_model(self, path="isolation_forest.pkl"):
        """Carga un modelo previamente entrenado desde un archivo."""
        try:
            self.isolation_forest = joblib.load(path)
            self.model_trained = True
            print(f"Modelo cargado desde {path}")
            return True
        except FileNotFoundError:
            print("No se encontró el modelo. Entrena uno nuevo.")
            return False
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar el sistema de seguridad
    security = SecuritySystem()
    
    # Agregar honeypot
    login_honeypot = security.add_honeypot("login_form")
    
    # Simular algunas interacciones normales
    for i in range(20):
        interaction = {
            "user_id": f"user_{i}",
            "action": "login",
            "timestamp": 1615000000 + i*3600,
            "ip": f"192.168.1.{i % 255}",
            "user_agent": "Mozilla/5.0"
        }
        security.log_interaction(interaction)
    
    # Entrenar el modelo
    security.train_model()
    
    # Detectar una posible anomalía
    suspicious_interaction = {
        "user_id": "attacker",
        "action": "login",
        "timestamp": 1615000000,
        "ip": "10.0.0.1",
        "user_agent": "Suspicious/1.0"
    }
    
    is_safe = security.handle_security(suspicious_interaction, "attacker")
    print(f"¿La interacción es segura? {is_safe}")
    
    # Guardar el modelo entrenado
    security.save_model()
    
    print("Sistema de seguridad inicializado correctamente.")
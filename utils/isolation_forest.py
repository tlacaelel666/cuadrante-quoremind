import joblib
from sklearn.ensemble import IsolationForest


class Honeypot:
    def __init__(self, name, threshold=0.5):
        self.name = name
        self.threshold = threshold
        self.interactions = []

    def record_interaction(self, interaction):
        """registra interacciones capturadas por el honeypot"""
        self.interactions.append(interaction)
        print(f"[{self.name}]interaccion registrada:{interaction}")

    def export_logs(self):
        """exporta los logs de interacciones"""
        return self.interactions


def save_model(self, path="isolation_forest.pkl"):
    joblib.dump(self.isolation_forest, path)
    print(f"Modelo guardado en {path}")


def load_model(self, path="isolation_forest.pkl"):
    try:
        self.isolation_forest = joblib.load(path)
        print(f"Modelo cargado desde {path}")
    except FileNotFoundError:
        print("No se encontró el modelo. Entrena uno nuevo.")


# Sistema de Seguridad con IA
class SecuritySystem:
    def __init__(self):
        self.honeypots = {}  # Honeypots para atrapar actividades sospechosas
        self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        self.interactions_log = []  # Registro de interacciones para entrenamiento

    def add_honeypot(self, name, honeypot):
        """Agrega un honeypot para monitorear actividades sospechosas."""
        self.honeypots[name] = honeypot

    @staticmethod
    def sanitize_input(interaction):
        """Sanitiza la entrada eliminando contenido malicioso o inválido."""
        sanitized_interaction = {k: (v if isinstance(v, (int, float, str)) else None) for k, v in interaction.items()}
        return sanitized_interaction

    def detect_anomalies(self, interaction):
        """Detecta anomalías usando Isolation Forest."""
        if not self.isolation_forest or not self.interactions_log:
            return False  # Sin modelo o datos previos, no se detectan anomalías.

        # Convertir la interacción a un vector numérico (ejemplo simplificado)
        interaction_vector = [hash(str(interaction[key])) % 1e5 for key in sorted(interaction.keys())]
        return self.isolation_forest.predict([interaction_vector])[0] == -1  # -1 indica anomalía

    @staticmethod
    def block_user(user_id):
        """Bloquea a un usuario identificado como malicioso."""
        print(f"Usuario {user_id} bloqueado por comportamiento sospechoso.")
        # Implementar la lógica de bloqueo (e.g., actualizar base de datos, marcar sesión, etc.)

    def train_model(self):
        """Entrena el modelo con las interacciones registradas."""
        if not self.interactions_log:
            print("No hay suficientes datos para entrenar el modelo.")
            return

        # Crear un conjunto de datos basado en el log de interacciones
        data = [[hash(str(interaction[key])) % 1e5 for key in sorted(interaction.keys())] for interaction in
                self.interactions_log]
        self.isolation_forest.fit(data)
        print("Modelo de seguridad entrenado con éxito.")

    def log_interaction(self, interaction):
        """Registra una interacción para monitoreo y entrenamiento futuro."""
        sanitized_interaction = self.sanitize_input(interaction)
        self.interactions_log.append(sanitized_interaction)

    def handle_security(self, interaction, user_id):
        """Sanitiza, registra y verifica la interacción."""
        sanitized_interaction = self.sanitize_input(interaction)
        self.log_interaction(sanitized_interaction)

        if self.detect_anomalies(sanitized_interaction):
            self.block_user(user_id)
            return False  # Indica que la interacción fue bloqueada

        return True  # Indica que la interacción es segura


print("security on")
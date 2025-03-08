
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
from bayes_logic import (
    BayesLogic,
    PRN,
    shannon_entropy,
    calculate_cosines  # Ajusta el import a tu conveniencia
)

# =============
#   Clase 1
# =============
class QuantumBayesMahalanobis(BayesLogic):
    def __init__(self):
        super().__init__()
        self.covariance_estimator = EmpiricalCovariance()

    def _get_inverse_covariance(self, data: np.ndarray) -> np.ndarray:
        """
        Ajusta el estimador de covarianza y retorna la inversa (o pseudo-inversa) de la matriz de covarianza.
        """
        self.covariance_estimator.fit(data)
        cov_matrix = self.covariance_estimator.covariance_
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
        return inv_cov_matrix

    def compute_quantum_mahalanobis(self, quantum_states_A: np.ndarray, quantum_states_B: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia de Mahalanobis entre cada estado en 'quantum_states_B' 
        respecto a la distribución de 'quantum_states_A'. Devuelve un arreglo 1D con las distancias.
        """
        inv_cov_matrix = self._get_inverse_covariance(quantum_states_A)
        mean_A = np.mean(quantum_states_A, axis=0)

        # Vectorización de la distancia de Mahalanobis
        diff_B = quantum_states_B - mean_A  # shape: (n_samples_B, n_dims)
        # dist^2 = (x - µ)^T * inv(COV) * (x - µ)
        # Se puede computar con np.einsum o multiplicación adecuada
        aux = diff_B @ inv_cov_matrix  # (n_samples_B, n_dims)
        dist_sqr = np.einsum('ij,ij->i', aux, diff_B)  # Producto elemento a elemento y suma por fila
        distances = np.sqrt(dist_sqr)  # (n_samples_B,)
        return distances

    def quantum_cosine_projection(self, quantum_states: np.ndarray, entropy: float, coherence: float) -> tf.Tensor:
        """
        Proyecta los estados cuánticos usando cosenos directores y 
        calcula la distancia de Mahalanobis entre dos proyecciones vectorizadas.
        Devuelve las distancias normalizadas (softmax).
        """
        cos_x, cos_y, cos_z = calculate_cosines(entropy, coherence)

        # Empleamos broadcasting:
        #   - Proyección A: [cos_x, cos_y]
        #   - Proyección B: [cos_x * cos_z, cos_y * cos_z]
        # States shape: (N, 2) => Multiplicamos cada columna por el factor respectivo
        projected_states_A = quantum_states * np.array([cos_x, cos_y])
        projected_states_B = quantum_states * np.array([cos_x * cos_z, cos_y * cos_z])

        # Calculamos las distancias de Mahalanobis vectorizadas
        mahalanobis_distances = self.compute_quantum_mahalanobis(
            projected_states_A,
            projected_states_B
        )
        # Normalizamos las distancias con softmax (conversión a tensor)
        mahalanobis_distances_tf = tf.convert_to_tensor(mahalanobis_distances, dtype=tf.float32)
        normalized_distances = tf.nn.softmax(mahalanobis_distances_tf)
        return normalized_distances

    def calculate_quantum_posterior_with_mahalanobis(self,
                                                     quantum_states: np.ndarray,
                                                     entropy: float,
                                                     coherence: float):
        """
        Calcula la probabilidad posterior considerando la distancia de Mahalanobis
        sobre proyecciones cuánticas.
        """
        quantum_projections = self.quantum_cosine_projection(
            quantum_states,
            entropy,
            coherence
        )

        # Convertir a tensor y calcular covarianza
        tensor_projections = tf.convert_to_tensor(quantum_projections, dtype=tf.float32)
        quantum_covariance = tfp.stats.covariance(tensor_projections, sample_axis=0)

        # Calcular prior cuántico basado en la traza de la covarianza
        # Dividimos la traza de la matriz entre su dimensión para mantenerla en [0, 1], aprox
        dim = tf.cast(tf.shape(quantum_covariance)[0], tf.float32)
        quantum_prior = tf.linalg.trace(quantum_covariance) / dim

        # Calcular prob posterior usando métodos de BayesLogic
        # 1) high_coherence_prior
        # 2) conditional_probability
        # 3) posterior_probability
        prior_coherence = self.calculate_high_coherence_prior(coherence)
        joint_prob = self.calculate_joint_probability(coherence, 1, tf.reduce_mean(tensor_projections))
        cond_prob = self.calculate_conditional_probability(joint_prob, quantum_prior)
        posterior = self.calculate_posterior_probability(quantum_prior, prior_coherence, cond_prob)

        return posterior, quantum_projections

    def predict_quantum_state(self,
                              quantum_states: np.ndarray,
                              entropy: float,
                              coherence: float):
        """
        Predice el siguiente estado cuántico basado en la proyección y la distancia de Mahalanobis.
        """
        posterior, projections = self.calculate_quantum_posterior_with_mahalanobis(
            quantum_states,
            entropy,
            coherence
        )

        # Sumar las proyecciones * posterior, para generar un "estado futuro"
        # Posterior es escalar, projections es un vector => expandimos posterior
        next_state_prediction = tf.reduce_sum(
            tf.multiply(projections, tf.expand_dims(posterior, -1)),
            axis=0
        )
        return next_state_prediction, posterior


# =============
#   Clase 2
# =============
class EnhancedPRN(PRN):
    """
    Extiende PRN para registrar distancias de Mahalanobis y con ello definir
    un 'ruido cuántico' adicional.
    """
    def __init__(self, influence: float = 0.5, algorithm_type: str = None, **parameters):
        """
        Se mantiene la firma de PRN, añadiendo almacenamiento de registros.
        """
        super().__init__(influence, algorithm_type, **parameters)
        self.mahalanobis_records = []

    def record_quantum_noise(self, probabilities: dict, quantum_states: np.ndarray):
        """
        Registra ruido considerando estados cuánticos y distancia de Mahalanobis,
        análogo a 'record_noise' de la superclase.
        """
        # Calculamos la entropía a partir de las 'probabilities'
        # (esta parte se asume que existe en la clase base con un método record_noise,
        #  en caso contrario, implementar la lógica adecuada.)
        entropy = self.record_noise(probabilities)

        # Calcular la distancia de Mahalanobis promedio
        cov_estimator = EmpiricalCovariance().fit(quantum_states)
        mean_state = np.mean(quantum_states, axis=0)
        inv_cov = np.linalg.pinv(cov_estimator.covariance_)

        # Vectorización para distancias
        diff = quantum_states - mean_state
        aux = diff @ inv_cov
        dist_sqr = np.einsum('ij,ij->i', aux, diff)
        distances = np.sqrt(dist_sqr)

        mahal_mean = np.mean(distances)
        self.mahalanobis_records.append(mahal_mean)

        return entropy, mahal_mean


# =============
#   Clase 3
# =============
class QuantumNoiseCollapse(QuantumBayesMahalanobis):
    """
    Combina la lógica bayesiana cuántica y el registro de PRN para simular colapso de onda
    con distancia de Mahalanobis.
    """
    def __init__(self):
        super().__init__()
        # Se inyecta un EnhancedPRN con un valor por defecto de influencia=0.5, p.ej.
        self.prn = EnhancedPRN(influence=0.5)

    def simulate_wave_collapse(self,
                               quantum_states: np.ndarray,
                               prn_influence: float,
                               previous_action: int):
        """
        Simula el colapso de onda con ruido cuántico integrando la distancia de Mahalanobis.
        """
        # Construimos diccionario de probabilidades a modo de ejemplo
        probabilities = {str(i): np.sum(state) for i, state in enumerate(quantum_states)}

        # Registra entropía y distancia
        entropy, mahalanobis_mean = self.prn.record_quantum_noise(probabilities, quantum_states)

        # Calcular cosenos directores con entropía y la distancia (usada como "coherence" base)
        cos_x, cos_y, cos_z = calculate_cosines(entropy, mahalanobis_mean)

        # Ejemplo: definimos coherencia a partir de la distancia de Mahalanobis y los cosenos
        coherence = np.exp(-mahalanobis_mean) * (cos_x + cos_y + cos_z) / 3.0

        # Usar el método de BayesLogic para integrar y decidir acción
        bayes_probs = self.calculate_probabilities_and_select_action(
            entropy=entropy,
            coherence=coherence,
            prn_influence=prn_influence,
            action=previous_action
        )

        # Proyectar estados y retornar colapso
        # (Opcional: se podría usar quantum_cosine_projection con entropía/coherence)
        projected_states = self.quantum_cosine_projection(
            quantum_states,
            entropy,
            coherence
        )
        # "Colapsamos" multiplicando la proyección por la acción
        collapsed_state = tf.reduce_sum(
            tf.multiply(
                projected_states,
                tf.cast(bayes_probs["action_to_take"], tf.float32)
            )
        )

        return {
            "collapsed_state": collapsed_state,
            "action": bayes_probs["action_to_take"],
            "entropy": entropy,
            "coherence": coherence,
            "mahalanobis_distance": mahalanobis_mean,
            "cosines": (cos_x, cos_y, cos_z)
        }

    def objective_function_with_noise(self,
                                      quantum_states: np.ndarray,
                                      target_state: np.ndarray,
                                      entropy_weight: float = 1.0) -> tf.Tensor:
        """
        Función objetivo que combina fidelidad, entropía y distancia de Mahalanobis.
        """
        # Calcular fidelidad cuántica (como ejemplo simple: |<ψ|φ>|^2)
        fidelity = tf.abs(tf.reduce_sum(quantum_states * tf.cast(target_state, quantum_states.dtype)))**2

        # Registrar ruido: entropía y distancia de Mahalanobis
        probabilities = {str(i): np.sum(st) for i, st in enumerate(quantum_states)}
        entropy, mahalanobis_dist = self.prn.record_quantum_noise(probabilities, quantum_states)

        # Combinar métrica
        # (1 - fidelidad) + a*entropía + (1 - e^{-Mahalanobis})
        objective_value = (1 - fidelity) \
                          + entropy_weight * entropy \
                          + (1 - np.exp(-mahalanobis_dist))

        return objective_value

    def optimize_quantum_state(self,
                               initial_states: np.ndarray,
                               target_state: np.ndarray,
                               max_iterations: int = 100):
        """
        Optimiza los estados cuánticos mediante descensos de gradiente simplificados.
        """
        # Convertir a tf.Variable para permitir gradientes
        current_states = tf.Variable(initial_states, dtype=tf.float32)
        best_objective = float('inf')
        best_states = None

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        for _ in range(max_iterations):
            with tf.GradientTape() as tape:
                objective = self.objective_function_with_noise(current_states.numpy(), target_state)
            grads = tape.gradient(objective, [current_states])

            # Si por alguna razón grads es None (estado no se ve afectado), romper
            if grads[0] is None:
                break

            optimizer.apply_gradients(zip(grads, [current_states]))

            # Re-evaluamos el objetivo tras el paso
            new_objective = self.objective_function_with_noise(current_states.numpy(), target_state)
            if new_objective < best_objective:
                best_objective = new_objective
                best_states = current_states.numpy().copy()

        return best_states, best_objective


# ====================
#     Ejemplo de uso
# ====================
if __name__ == "__main__":
    # Inicializar el sistema
    qnc = QuantumNoiseCollapse()

    # Estados cuánticos iniciales
    initial_states = np.array([
        [0.8, 0.2],
        [0.9, 0.4],
        [0.1, 0.7]
    ])

    # Estado objetivo
    target_state = np.array([1.0, 0.0])

    # Optimizar estados
    optimized_states, final_objective = qnc.optimize_quantum_state(
        initial_states,
        target_state
    )

    # Simular colapso final
    final_collapse = qnc.simulate_wave_collapse(
        optimized_states,
        prn_influence=0.5,
        previous_action=0
    )

    print("Optimized states:", optimized_states)
    print("Final objective value:", final_objective)
    print("Final collapse result:", final_collapse)

"""
--------------------------------------------------------------------------------

Resumen de funciones destacadas:

1. Cálculo vectorizado de la distancia de Mahalanobis.  
   – Se evita el bucle for en compute_quantum_mahalanobis, usando la operación (x - μ) @ inv_cov * (x - μ) y después la suma por eje.  

2. Uso de broadcasting en quantum_cosine_projection para la generación de projected_states_A y projected_states_B.  
   – Esto reemplaza la comprensión de listas al multiplicar por [cos_x, cos_y] y [cos_x*cos_z, cos_y*cos_z].  

3. Minimización vía Adam en optimize_quantum_state (o el optimizador de preferencia), manteniendo la posibilidad de gradientes en TensorFlow.  

4. Se introdujo el método _get_inverse_covariance() como helper para no repetir la misma lógica de ajustar la covarianza e invertirla.  

5. Uso de tf.Variable y GradientTape para hacer el “descenso de gradiente” sobre los estados cuánticos.  

"""
fecha: (apr)04/09/2025
Autor: Jacobo Tlacaelel Mina Rodríguez 
Versión: QuoreMind V1.1.3

python import numpy as np from scipy.stats import multivariate_normal from scipy.spatial import distance

--- Definición de funciones auxiliares ---
def get_angles_from_direction_cosines(dx, dy, dz): """ Obtiene ángulos theta y phi a partir de cosenos directores (dx, dy, dz). """ try: theta = np.arccos(dz) phi = np.arctan2(dy, dx) return theta, phi except Exception as e: raise ValueError("Error al calcular ángulos desde cosenos directores.") from e

def get_amplitudes_from_angles(theta, phi): """ Obtiene amplitudes de qubit a partir de ángulos theta y phi. """ try: amplitude_0 = np.cos(theta / 2) amplitude_1 = np.sin(theta / 2) * np.cos(phi) + 1j * np.sin(theta / 2) * np.sin(phi) return np.array([amplitude_0, amplitude_1]) except Exception as e: raise ValueError("Error al calcular amplitudes desde ángulos.") from e

def get_direction_cosines_from_angles(theta, phi): """ Calcula los cosenos directores a partir de ángulos theta y phi. """ try: dx = np.sin(theta) * np.cos(phi) dy = np.sin(theta) * np.sin(phi) dz = np.cos(theta) return dx, dy, dz except Exception as e: raise ValueError("Error al calcular cosenos directores desde ángulos.") from e

def apply_hadamard(amplitudes): """ Aplica la puerta Hadamard a un estado cuántico. """ try: H = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]]) return np.dot(H, amplitudes) except Exception as e: raise ValueError("Error al aplicar la puerta Hadamard.") from e

def function_of_amplitudes_and_parameter(amplitudes, parameter): """ Función auxiliar para relacionar amplitudes con un parámetro. """ try: observable = np.array([[parameter, 0], [0, -parameter]]) return np.real(np.dot(amplitudes.conjugate(), np.dot(observable, amplitudes))) except Exception as e: raise ValueError("Error al calcular función de amplitudes y parámetro.") from e

def simulate_measurement(amplitudes, true_parameter): """ Simula una medición que da información sobre el parámetro. """ try: noise = np.random.normal(0, 0.1) measurement = function_of_amplitudes_and_parameter(amplitudes, true_parameter) + noise return measurement except Exception as e: raise ValueError("Error al simular la medición.") from e

def probability_density_function(measurement, parameter, quantum_state_amplitudes): """ Calcula la función de densidad de probabilidad de una medición dado un parámetro. """ try: mean = function_of_amplitudes_and_parameter(quantum_state_amplitudes, parameter) std_dev = 0.1 return (1/(std_devnp.sqrt(2np.pi))) * np.exp(-0.5*((measurement - mean)/std_dev)**2) except Exception as e: raise ValueError("Error al calcular la función de densidad de probabilidad.") from e

def likelihood(parameter, measurement, quantum_state_amplitudes): """ Calcula la verosimilitud (likelihood) de un parámetro dado una medición. """ try: return probability_density_function(measurement, parameter, quantum_state_amplitudes) except Exception as e: raise ValueError("Error al calcular la verosimilitud.") from e

def update_bayesian_estimate(prior_mean, prior_variance, measurement, quantum_state_amplitudes, num_samples=1000): """ Actualiza la estimación bayesiana de un parámetro. """ try: parameter_samples = np.random.normal(prior_mean, np.sqrt(prior_variance), num_samples) likelihoods = np.array([likelihood(param, measurement, quantum_state_amplitudes) for param in parameter_samples]) weights = likelihoods / np.sum(likelihoods) posterior_mean = np.sum(weights * parameter_samples) posterior_variance = np.sum(weights * (parameter_samples - posterior_mean)**2) return posterior_mean, posterior_variance except Exception as e: raise ValueError("Error al actualizar la estimación bayesiana.") from e

def complex_to_real_vector(complex_vector): """ Convierte un vector complejo a su representación real. """ try: return np.concatenate([complex_vector.real, complex_vector.imag]) except Exception as e: raise ValueError("Error al convertir vector complejo a real.") from e

--- Inicialización ---
planck_constant_reduced = 1.054571817e-34 # ħ (h-bar) en J·s

print("=== Manipulación de Amplitudes Cuánticas y Gestión Conceptual ===")

Definir un estado cuántico inicial (ejemplo: qubit)
print("\n--- Inicialización del Estado Cuántico ---")

Opción 1: Usando amplitudes directamente
quantum_state_amplitudes = np.array([1.0, 0.0], dtype=complex) # Estado |0> print(f"Estado inicial (amplitudes): {quantum_state_amplitudes}")

Opción 2: Usando cosenos directores
initial_dx, initial_dy, initial_dz = 1.0, 0.0, 0.0 # Corresponde a estado sobre eje x print(f"Cosenos directores iniciales: ({initial_dx}, {initial_dy}, {initial_dz})") initial_theta, initial_phi = get_angles_from_direction_cosines(initial_dx, initial_dy, initial_dz) print(f"Ángulos correspondientes: theta = {initial_theta:.4f}, phi = {initial_phi:.4f}")

quantum_state_amplitudes = get_amplitudes_from_angles(initial_theta, initial_phi) print(f"Estado inicial desde cosenos directores: {quantum_state_amplitudes}")

--- Manipulación de Amplitudes ---
print("\n--- Manipulación de Amplitudes ---")

Aplicamos puerta Hadamard
print("Aplicando puerta Hadamard...") quantum_state_amplitudes = apply_hadamard(quantum_state_amplitudes) print(f"Estado después de Hadamard: {quantum_state_amplitudes}")

--- Metodología Bayesiana ---
print("\n--- Metodología Bayesiana ---")

Prior (creencia inicial sobre el parámetro)
prior_mean = 0.0 prior_variance = 1.0 print(f"Prior: N({prior_mean}, {prior_variance})")

Simulamos una medición
true_parameter = 0.5 # Valor "real" del parámetro (desconocido en aplicaciones reales) observed_measurement = simulate_measurement(quantum_state_amplitudes, true_parameter) print(f"Medición observada: {observed_measurement:.4f}")

Actualización Bayesiana
posterior_mean, posterior_variance = update_bayesian_estimate( prior_mean, prior_variance, observed_measurement, quantum_state_amplitudes) print(f"Posterior: N({posterior_mean:.4f}, {posterior_variance:.4f})") print(f"Valor real del parámetro: {true_parameter}")

--- Distancias ---
print("\n--- Cálculo de Distancias ---")

Definimos algunos estados de referencia
reference_states = [ np.array([1.0, 0.0], dtype=complex), # |0> np.array([0.0, 1.0], dtype=complex), # |1> np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex), # |+> np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex) # |-> ] reference_states_names = ["|0>", "|1>", "|+>", "|->"]

Convertimos a arrays reales para el cálculo de distancias
real_reference_states = np.array([complex_to_real_vector(state) for state in reference_states]) real_current_state = complex_to_real_vector(quantum_state_amplitudes)

Distancia de Mahalanobis
print("\n--- Distancia de Mahalanobis ---") mean_reference = np.mean(real_reference_states, axis=0) covariance_matrix = np.cov(real_reference_states.T)

try: inverse_covariance = np.linalg.inv(covariance_matrix) mahalanobis_distance = distance.mahalanobis( real_current_state, mean_reference, inverse_covariance) print(f"Distancia de Mahalanobis al centroide de referencia: {mahalanobis_distance:.4f}") except np.linalg.LinAlgError: print("La matriz de covarianza no es invertible.") inverse_covariance = np.linalg.pinv(covariance_matrix) mahalanobis_distance = distance.mahalanobis( real_current_state, mean_reference, inverse_covariance) print(f"Distancia de Mahalanobis (usando pseudoinversa): {mahalanobis_distance:.4f}")

Distancia de Minkowski para diferentes valores de p
print("\n--- Distancias de Minkowski ---") p_values = [1, 2, np.inf] # Manhattan, Euclídea, Chebyshev for p in p_values: for i, ref_state in enumerate(reference_states): real_ref = complex_to_real_vector(ref_state) minkowski_dist = distance.minkowski(real_current_state, real_ref, p=p) p_name = {1: "Manhattan", 2: "Euclídea", np.inf: "Chebyshev"}.get(p, f"Minkowski-{p}") print(f"Distancia {p_name} al estado {reference_states_names[i]}: {minkowski_dist:.4f}")

--- Gestión Conceptual ---
print("\n--- Gestión Conceptual ---")

Calculamos fidelidad con diferentes estados objetivo
print("Fidelidades con estados de referencia:") for i, ref_state in enumerate(reference_states): overlap = np.abs(np.dot(quantum_state_amplitudes.conjugate(), ref_state))**2 print(f"Fidelidad con {reference_states_names[i]}: {overlap:.4f}")

Decisión basada en fidelidad
fidelity_threshold = 0.9 max_fidelity = 0 max_fidelity_index = -1

for i, ref_state in enumerate(reference_states): overlap = np.abs(np.dot(quantum_state_amplitudes.conjugate(), ref_state))**2 if overlap > max_fidelity: max_fidelity = overlap max_fidelity_index = i

print(f"\nMáxima fidelidad: {max_fidelity:.4f} con estado {reference_states_names[max_fidelity_index]}")

if max_fidelity < fidelity_threshold: print("El estado actual está por debajo del umbral de fidelidad.") print("Aplicando operación de corrección...") target_state = reference_states[max_fidelity_index] print(f"Ajustando hacia el estado {reference_states_names[max_fidelity_index]}...") else: print(f"El estado actual cumple con el umbral de fidelidad con {reference_states_names[max_fidelity_index]}.")

"""
Terms | Privacy
© SmokAppStore OnStartups
"""
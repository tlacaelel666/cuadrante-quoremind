from typing import Tuple, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
from logic import BayesLogic, shannon_entropy, calculate_cosines # Asegúrate de que logic.py esté en el mismo directorio

# --- Funciones auxiliares para la visualización de la red neuronal (simplificadas) ---
def initialize_node():
    """Inicializa un nodo de la red neuronal (placeholder)."""
    return {'active': False} # Simplificado: inicialmente no activo

def is_active(node):
    """Verifica si un nodo está activo (placeholder)."""
    return node['active']

# --- Clase TimeSeries Corregida ---
class TimeSeries:
    def __init__(self, amplitud: float, frecuencia: float, fase: float):
        """Inicializa la serie temporal sinusoidal."""
        self.amplitud = amplitud
        self.frecuencia = frecuencia
        self.fase = fase

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evalúa la onda sinusoidal en los puntos x."""
        return self.amplitud * np.sin(2 * np.pi * self.frecuencia * x + self.fase)

    def get_phase(self) -> float:
        """Retorna la fase actual de la onda."""
        return self.fase

    def set_phase(self, nueva_fase: float) -> None:
        """Establece una nueva fase para la onda."""
        self.fase = nueva_fase

# --- Función colapso_onda Corregida y Movida fuera de TimeSeries ---
def colapso_onda(onda_superpuesta: np.ndarray, bayes_logic: BayesLogic, prn_influence: float, previous_action: int) -> Tuple[float, int]:
    """
    Simula el colapso de la onda superpuesta basándose en la lógica Bayesiana.

    Args:
        onda_superpuesta (np.ndarray): Array numpy representando la onda superpuesta.
        bayes_logic (BayesLogic): Instancia de la clase BayesLogic para la toma de decisiones.
        prn_influence (float): Influencia del Ruido Probabilístico de Referencia (PRN).
        previous_action (int): Acción previa tomada (0 o 1).

    Returns:
        Tuple[float, int]: Estado colapsado (fase en radianes) y la acción seleccionada (0 o 1).
    """
    # 1. Calcular la entropía de la onda superpuesta
    entropy = shannon_entropy(onda_superpuesta)

    # 2. Definir un valor ambiental para calcular los cosenos directores (ejemplo)
    env_value = 0.8

    # 3. Calcular los cosenos directores basados en la entropía y el valor ambiental
    cos_x, cos_y, cos_z = calculate_cosines(entropy, env_value)

    # 4. Calcular la coherencia basada en los cosenos directores (ejemplo)
    coherence = (cos_x + cos_y + cos_z) / 3

    # 5. Utilizar BayesLogic para determinar la acción a tomar basándose en la entropía y coherencia
    probabilities = bayes_logic.calculate_probabilities_and_select_action(
        entropy=entropy,
        coherence=coherence,
        prn_influence=prn_influence,
        action=previous_action
    )

    action_to_take = probabilities["action_to_take"]

    # 6. Simular el estado colapsado basándose en la acción
    if action_to_take == 1:
        estado_colapsado = np.pi  # Colapso hacia una fase de 180 grados (ejemplo)
    else:
        estado_colapsado = 0.5  # Colapso hacia una fase de 0.5 radianes (ejemplo)

    return estado_colapsado, action_to_take


# --- Funciones de visualización ---
def visualize_wave_and_network(network: List[List[Dict[str, bool]]], iteration: int, t: float, estado_colapsado_fase: float = None) -> None:
    """
    Visualiza el estado de la red neuronal y la función de onda, incluyendo el estado colapsado.

    Args:
        network (List[List[Dict[str, bool]]]): La red neuronal a visualizar.
        iteration (int): El número de iteración actual.
        t (float): El tiempo actual para la función de onda.
        estado_colapsado_fase (float, opcional): La fase del estado colapsado para visualizar.
    """
    # 1. Gráfico de la Función de Onda
    x_wave = np.linspace(0, 10, 500)  # Rango de x ajustado según necesidad
    onda_incidente = TimeSeries(amplitud=0.5, frecuencia=1.5, fase=-np.pi / 21) # Re-instancia ondas para graficar
    onda_reflejada = TimeSeries(amplitud=0.5, frecuencia=1.5, fase=-np.pi / 21 + np.pi)
    y_incidente = onda_incidente.evaluate(x_wave)
    y_reflejada = onda_reflejada.evaluate(x_wave)
    y_superpuesta = y_incidente + y_reflejada
    y_colapsada = TimeSeries(amplitud=0.5, frecuencia=1.5, fase=estado_colapsado_fase).evaluate(x_wave) if estado_colapsado_fase is not None else None


    plt.figure(figsize=(14, 7))  # Figura más grande para ambos gráficos
    plt.subplot(1, 2, 1)  # Subplot para la onda

    plt.plot(x_wave, y_incidente, label="Onda Incidente", color="blue", alpha=0.6)
    plt.plot(x_wave, y_reflejada, label="Onda Reflejada", color="red", alpha=0.6)
    plt.plot(x_wave, y_superpuesta, label="Onda Superpuesta (Antes del Colapso)", color="green")
    if y_colapsada is not None:
        plt.plot(x_wave, y_colapsada, label=f"Onda Colapsada (Fase: {np.degrees(estado_colapsado_fase):.0f}°)", color="purple", linestyle='--')

    plt.xlabel("x")
    plt.ylabel("ψ(x)")
    plt.title(f"Superposición y Colapso de Ondas en t={t:.2f}")
    plt.grid(True)
    plt.legend()

    # 2. Gráfico del Estado de la Red Neuronal
    plt.subplot(1, 2, 2)  # Subplot para la red
    for layer_index, layer in enumerate(network):
        for node_index, node in enumerate(layer):
            if is_active(node):
                plt.scatter(layer_index, node_index, color='red', marker='o', s=100) # s aumenta el tamaño de los puntos

    plt.title(f"Estado de la Red Neuronal en Iteración {iteration}")
    plt.xlabel("Índice de Capa")
    plt.ylabel("Índice de Nodo")
    plt.xlim(-1, len(network))
    plt.ylim(-1, max(len(layer) for layer in network) if network else 0) # Manejo de red vacía
    plt.grid(True)

    plt.tight_layout()  # Ajusta los parámetros del subplot para un diseño apretado
    plt.show()


# --- Función principal para ejecutar la simulación ---
def main():
    """Función principal para ejecutar la simulación del colapso de onda y red neuronal."""
    # Parámetros de las ondas
    amplitud = 0.5
    frecuencia = 1.5
    fase_inicial = -np.pi / 21

    # Crear las ondas iniciales
    onda_incidente = TimeSeries(amplitud, frecuencia, fase_inicial)
    onda_reflejada = TimeSeries(amplitud, frecuencia, fase_inicial + np.pi)

    # Instanciar BayesLogic y PRN
    bayes_logic = BayesLogic()
    prn = PRN(influence=0.5)

    # Acción previa inicial
    previous_action = 1

    # Inicializar una red neuronal muy simple (solo para visualización)
    network = [[initialize_node() for _ in range(n)] for n in [2, 3, 2, 2]]

    # Simulación iterativa
    for iteration in range(10):
        t = iteration * 0.5  # Tiempo para la función de onda

        # Generar puntos x y onda superpuesta en cada iteración
        x = np.linspace(0, 10, 500)
        y_incidente = onda_incidente.evaluate(x)
        y_reflejada = onda_reflejada.evaluate(x)
        y_superpuesta = y_incidente + y_reflejada

        # Simular el colapso de la onda y obtener la acción y el estado colapsado
        estado_colapsado_fase, selected_action = colapso_onda(y_superpuesta, bayes_logic, prn.influence, previous_action)

        # Establecer la fase de la onda incidente (como ejemplo de "influencia del colapso")
        onda_incidente.set_phase(estado_colapsado_fase) # La fase incidente se ajusta según el "colapso"
        previous_action = selected_action # Actualizar la acción previa para la siguiente iteración

        # Simular la activación de algunos nodos de la red neuronal (basado en la acción - ejemplo)
        for layer_index, layer in enumerate(network):
            for node_index, node in enumerate(layer):
                if np.random.rand() < (0.2 if selected_action == 1 else 0.1): # Probabilidad de activación dependiente de la acción
                    network[layer_index][node_index]['active'] = True
                else:
                    network[layer_index][node_index]['active'] = False

        # Visualizar la onda y la red neuronal en cada iteración, pasando el estado colapsado para graficar
        visualize_wave_and_network(network, iteration, t, estado_colapsado_fase)

        # Imprimir información de la iteración (opcional)
        entropy_val = shannon_entropy(y_superpuesta)
        cos_x_val, cos_y_val, cos_z_val = calculate_cosines(entropy_val, 0.8)
        coherence_val = (cos_x_val + cos_y_val + cos_z_val) / 3

        print(f"Iteración {iteration}:")
        print(f"  Entropía de Shannon: {entropy_val:.4f}")
        print(f"  Cosenos Directores: cos_x = {cos_x_val:.4f}, cos_y = {cos_y_val:.4f}, cos_z = {cos_z_val:.4f}")
        print(f"  Coherencia: {coherence_val:.4f}")
        print(f"  Acción seleccionada: {'Mover Derecha' if selected_action == 1 else 'Mover Izquierda'}")
        print(f"  Estado colapsado (fase): {np.degrees(estado_colapsado_fase):.2f} grados\n")


if __name__ == "__main__":
    main()

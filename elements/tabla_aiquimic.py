#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Tuple, Optional
from qiskit import Aer, execute, QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

class ElementQuantumMomentum:
    """
    Clase para modelar elementos químicos como circuitos cuánticos
    y calcular sus distribuciones de momentum características.
    """
    
    def __init__(self):
        """Inicializa la biblioteca de elementos químicos con sus configuraciones cuánticas."""
        # Diccionario que mapea símbolos de elementos a sus configuraciones
        self.elements_config = {
            "H": {
                "name": "Hidrógeno",
                "qubits": 1,
                "electron_config": "1s¹",
                "properties": ["Enlaces simples", "Reducción", "Gas ligero"]
            },
            "He": {
                "name": "Helio",
                "qubits": 2,
                "electron_config": "1s²",
                "properties": ["Inerte", "Gas noble", "No reactivo"]
            },
            "Li": {
                "name": "Litio",
                "qubits": 3,
                "electron_config": "1s² 2s¹",
                "properties": ["Altamente reactivo", "Metal alcalino", "Donador de electrones"]
            },
            "C": {
                "name": "Carbono",
                "qubits": 4,  # Simplificado (en realidad necesitaría más)
                "electron_config": "1s² 2s² 2p²",
                "properties": ["Tetravalencia", "Estructuras complejas", "Base de química orgánica"]
            },
            "N": {
                "name": "Nitrógeno",
                "qubits": 4,  # Simplificado
                "electron_config": "1s² 2s² 2p³",
                "properties": ["Trivalente", "Componente de aminoácidos", "Gas diatómico"]
            },
            "O": {
                "name": "Oxígeno",
                "qubits": 4,  # Simplificado
                "electron_config": "1s² 2s² 2p⁴",
                "properties": ["Oxidante", "Divalente", "Esencial para la vida"]
            }
        }
    
    def initialize_element_circuit(self, symbol: str) -> Optional[QuantumCircuit]:
        """
        Crea un circuito cuántico que representa la estructura electrónica del elemento.
        
        Args:
            symbol: Símbolo químico del elemento
            
        Returns:
            Circuito cuántico que representa el elemento, o None si el elemento no está en la base de datos
        """
        if symbol not in self.elements_config:
            return None
            
        config = self.elements_config[symbol]
        num_qubits = config["qubits"]
        
        # Crear circuito
        qc = QuantumCircuit(num_qubits, name=f"{symbol}_{config['name']}")
        
        # Configurar el circuito según el elemento
        if symbol == "H":
            # Hidrógeno: Solo un qubit en superposición (puerta H)
            qc.h(0)
            
        elif symbol == "He":
            # Helio: Dos qubits en estado |11> (capa llena)
            qc.x(0)
            qc.x(1)
            
        elif symbol == "Li":
            # Litio: Dos primeros qubits en |11> (capa interna) y el tercero en superposición
            qc.x(0)
            qc.x(1)
            qc.h(2)  # El electrón de valencia en superposición
            
        elif symbol == "C":
            # Carbono: Simulación simplificada con 4 qubits
            # Los primeros dos qubits representan la capa interna completa
            qc.x(0)
            qc.x(1)
            # Los otros dos representan los 4 electrones de valencia en superposición
            qc.h(2)
            qc.h(3)
            # Entrelazamos para simular la tetravalencia
            qc.cx(2, 3)
            
        elif symbol == "N":
            # Nitrógeno: Similar al carbono pero con diferente configuración de valencia
            qc.x(0)
            qc.x(1)
            qc.h(2)
            qc.h(3)
            # Configuración diferente para representar los 3 enlaces posibles
            qc.cx(2, 3)
            qc.t(3)  # Fase adicional para diferenciar de C
            
        elif symbol == "O":
            # Oxígeno: Similar pero con configuración para los 2 enlaces típicos
            qc.x(0)
            qc.x(1)
            qc.h(2)
            qc.h(3)
            # Configuración para representar los 2 enlaces típicos
            qc.cx(2, 3)
            qc.s(3)  # Fase diferente a N
        
        return qc
    
    def get_momentum_distribution(self, element_symbol: str) -> Dict[str, float]:
        """
        Calcula la distribución de momentum para un elemento químico.
        
        Args:
            element_symbol: Símbolo del elemento químico
            
        Returns:
            Diccionario con las probabilidades de momentum
        """
        # Inicializar el circuito del elemento
        element_circuit = self.initialize_element_circuit(element_symbol)
        if element_circuit is None:
            return {}
            
        # Aplicar la QFT para obtener la representación de momentum
        num_qubits = element_circuit.num_qubits
        qft = QFT(num_qubits, inverse=False, do_swaps=True)
        
        # Componer el circuito con la QFT
        complete_circuit = element_circuit.compose(qft)
        
        # Simular el circuito
        simulator = Aer.get_backend('statevector_simulator')
        job = execute(complete_circuit, simulator)
        result = job.result()
        final_state = result.get_statevector(complete_circuit)
        
        # Calcular probabilidades
        probabilities = {format(i, '0' + str(num_qubits) + 'b'): np.abs(final_state[i])**2
                         for i in range(2**num_qubits)}
        
        # Filtrar solo probabilidades significativas
        return {k: v for k, v in probabilities.items() if v > 0.01}
    
    def generate_momentum_table(self, elements: List[str] = None) -> List[Dict]:
        """
        Genera una tabla de momentum cuántico para los elementos especificados.
        
        Args:
            elements: Lista de símbolos de elementos. Si es None, usa todos los elementos disponibles.
            
        Returns:
            Lista de diccionarios con la información de momentum para cada elemento
        """
        if elements is None:
            elements = list(self.elements_config.keys())
            
        momentum_table = []
        
        for symbol in elements:
            if symbol not in self.elements_config:
                continue
                
            # Obtener distribución de momentum
            momentum_dist = self.get_momentum_distribution(symbol)
            
            # Crear entrada para la tabla
            entry = {
                "element": symbol,
                "name": self.elements_config[symbol]["name"],
                "qubits": self.elements_config[symbol]["qubits"],
                "electron_config": self.elements_config[symbol]["electron_config"],
                "properties": self.elements_config[symbol]["properties"],
                "momentum_distribution": momentum_dist
            }
            
            momentum_table.append(entry)
            
        return momentum_table
    
    def visualize_momentum(self, element_symbol: str) -> None:
        """
        Visualiza la distribución de momentum para un elemento.
        
        Args:
            element_symbol: Símbolo del elemento
        """
        if element_symbol not in self.elements_config:
            print(f"Elemento {element_symbol} no encontrado en la base de datos.")
            return
            
        # Obtener distribución de momentum
        momentum_dist = self.get_momentum_distribution(element_symbol)
        
        # Crear gráfico
        element_name = self.elements_config[element_symbol]["name"]
        plt.figure(figsize=(10, 6))
        plt.bar(momentum_dist.keys(), momentum_dist.values())
        plt.title(f"Distribución de Momentum para {element_symbol} ({element_name})")
        plt.xlabel("Estados de Momentum")
        plt.ylabel("Probabilidad")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def print_momentum_table(self, elements: List[str] = None) -> None:
        """
        Imprime una tabla formateada con la información de momentum para los elementos especificados.
        
        Args:
            elements: Lista de símbolos de elementos. Si es None, usa todos los elementos disponibles.
        """
        if elements is None:
            elements = list(self.elements_config.keys())
            
        print("TABLA DE MOMENTUM CUÁNTICO PARA ELEMENTOS")
        print("=" * 80)
        print(f"{'Elemento':<8} {'Nombre':<12} {'Qubits':<6} {'Config. Electrónica':<20} {'Estados de Momentum Significativos'}")
        print("-" * 80)
        
        for symbol in elements:
            if symbol not in self.elements_config:
                continue
                
            config = self.elements_config[symbol]
            momentum_dist = self.get_momentum_distribution(symbol)
            
            # Formatear los estados de momentum como string
            momentum_states = ", ".join([f"|{state}> ({prob:.2f})" 
                                         for state, prob in momentum_dist.items()])
            
            print(f"{symbol:<8} {config['name']:<12} {config['qubits']:<6} {config['electron_config']:<20} {momentum_states}")
            
        print("=" * 80)
        print("NOTA: Esta tabla muestra la correspondencia teórica entre elementos químicos")
        print("      y sus distribuciones de momentum cuántico después de aplicar QFT.")
        print("      Las distribuciones pueden correlacionarse con propiedades químicas.")
    
    def analyze_momentum_patterns(self, elements: List[str] = None) -> Dict:
        """
        Analiza patrones en las distribuciones de momentum y busca correlaciones con propiedades químicas.
        
        Args:
            elements: Lista de símbolos de elementos a analizar
            
        Returns:
            Diccionario con análisis de patrones encontrados
        """
        if elements is None:
            elements = list(self.elements_config.keys())
            
        patterns = {}
        
        # Ejemplo de patrón: Uniformidad de la distribución
        patterns["uniformity"] = {}
        
        for symbol in elements:
            if symbol not in self.elements_config:
                continue
                
            momentum_dist = self.get_momentum_distribution(symbol)
            
            # Calcular la uniformidad (entropía normalizada)
            probs = list(momentum_dist.values())
            if probs:
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                max_entropy = np.log2(len(probs))
                uniformity = entropy / max_entropy if max_entropy > 0 else 1.0
                
                patterns["uniformity"][symbol] = uniformity
        
        return patterns

def main():
    """Función principal para demostrar el uso de la tabla de momentum cuántico."""
    quantum_elements = ElementQuantumMomentum()
    
    # Elementos fundamentales para la química orgánica
    basic_elements = ["H", "C", "N", "O"]
    
    print("\n1. Generando tabla de momentum para elementos fundamentales:")
    quantum_elements.print_momentum_table(basic_elements)
    
    print("\n2. Análisis detallado del Hidrógeno:")
    h_circuit = quantum_elements.initialize_element_circuit("H")
    if h_circuit:
        print(f"Circuito cuántico para H (Hidrógeno):")
        print(h_circuit.draw())
        
    print("\nDistribución de momentum para H:")
    h_momentum = quantum_elements.get_momentum_distribution("H")
    for state, prob in h_momentum.items():
        print(f"Estado |{state}>: Probabilidad = {prob:.6f}")
    
    print("\n3. Análisis de patrones de momentum:")
    patterns = quantum_elements.analyze_momentum_patterns(basic_elements)
    print("Uniformidad de las distribuciones de momentum:")
    for element, uniformity in patterns["uniformity"].items():
        print(f"{element}: {uniformity:.4f}")
    
    print("\n4. Tabla completa de momentum cuántico:")
    momentum_table = quantum_elements.generate_momentum_table()
    print(f"Generada tabla con {len(momentum_table)} elementos.")
    
    print("\nNOTA: Este modelo es una aproximación teórica que establece")
    print("una correspondencia entre elementos químicos y circuitos cuánticos.")
    print("Las distribuciones de momentum resultantes podrían correlacionarse")
    print("con propiedades químicas fundamentales, creando una 'tabla periódica cuántica'.")

if __name__ == "__main__":
    main()
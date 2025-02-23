# main.py

# 1. Importación de Módulos Necesarios
import numpy as np
import tensorflow as tf
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Importar módulos específicos del proyecto
import ibm_setup_conf
import circuito_principal
import modelo_hibrido

def main():
    # 2. Configuración del Entorno Cuántico
    backend = ibm_setup_conf.configurar_entorno()
    
    # 3. Construcción del Circuito Cuántico
    circuito = circuito_principal.crear_circuito()
    
    # 4. Definición y Entrenamiento del Modelo Híbrido
    modelo = modelo_hibrido.ModeloHibrido()
    datos_entrenamiento = obtener_datos_entrenamiento()  # Definir esta función según tus necesidades
    modelo.entrenar(datos_entrenamiento)
    
    # 5. Ejecución del Circuito en el Backend Seleccionado
    job = execute(circuito, backend, shots=1024)
    resultados = job.result()
    conteos = resultados.get_counts(circuito)
    
    # 6. Procesamiento y Visualización de Resultados
    print("Resultados de la ejecución del circuito:")
    print(conteos)
    plot_histogram(conteos)
    plt.show()

if __name__ == "__main__":
    main()

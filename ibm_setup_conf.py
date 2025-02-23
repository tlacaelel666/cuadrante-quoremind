#Marco de trabajo Coremind powered with Gemini AI 
Iniciar entorno 
#!/usr/bin/env python3
"""
ibm_quantum_cli.py - Script para interactuar con IBM Quantum desde la línea de comandos
Uso: python ibm_quantum_cli.py --token TU_TOKEN [--action {status,execute,list}]
"""

import argparse
import json
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.tools.monitor import job_monitor
import sys

def setup_api(token):
    """Configura la conexión con IBM Quantum."""
    try:
        IBMQ.save_account(token, overwrite=True)
        IBMQ.load_account()
        return IBMQ.get_provider('ibm-q')
    except Exception as e:
        print(f"Error al configurar la API: {str(e)}")
        sys.exit(1)

def list_backends(provider):
    """Lista todos los backends disponibles."""
    try:
        print("\nBackends disponibles:")
        for backend in provider.backends():
            status = backend.status()
            print(f"\nNombre: {backend.name()}")
            print(f"Estado: {'Operativo' if status.operational else 'No operativo'}")
            print(f"Cola de trabajos: {status.pending_jobs}")
            print(f"Qubits: {len(backend.properties().qubits)}")
    except Exception as e:
        print(f"Error al listar backends: {str(e)}")

def check_status(provider, backend_name=None):
    """Verifica el estado de un backend específico o todos."""
    try:
        if backend_name:
            backend = provider.get_backend(backend_name)
            status = backend.status()
            properties = backend.properties()
            print(f"\nEstado de {backend_name}:")
            print(f"Operativo: {status.operational}")
            print(f"Cola de trabajos: {status.pending_jobs}")
            print(f"Tiempo de coherencia T1 promedio: {sum(q.T1 for q in properties.qubits)/len(properties.qubits)} μs")
        else:
            list_backends(provider)
    except Exception as e:
        print(f"Error al verificar estado: {str(e)}")

def execute_circuit(provider, backend_name):
    """Ejecuta un circuito cuántico simple."""
    try:
        # Crear circuito simple de prueba
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)  # Puerta Hadamard en qubit 0
        circuit.cx(0, 1)  # CNOT controlado por qubit 0, objetivo qubit 1
        circuit.measure_all()

        # Ejecutar el circuito
        backend = provider.get_backend(backend_name)
        job = execute(circuit, backend=backend, shots=1024)
        
        print(f"Trabajo enviado al backend {backend_name}")
        print("ID del trabajo:", job.job_id())
        print("\nMonitoreando el trabajo...")
        job_monitor(job)
        
        # Obtener y mostrar resultados
        result = job.result()
        counts = result.get_counts()
        print("\nResultados:")
        print(json.dumps(counts, indent=2))
        
        return job.job_id()
    except Exception as e:
        print(f"Error al ejecutar circuito: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='CLI para IBM Quantum')
    parser.add_argument('--token', required=True, help='Tu token de API de IBM Quantum')
    parser.add_argument('--action', choices=['status', 'execute', 'list'], 
                      default='status', help='Acción a realizar')
    parser.add_argument('--backend', help='Nombre del backend específico')
    
    args = parser.parse_args()
    
    # Configurar la conexión
    provider = setup_api(args.token)
    
    # Ejecutar la acción solicitada
    if args.action == 'status':
        check_status(provider, args.backend)
    elif args.action == 'list':
        list_backends(provider)
    elif args.action == 'execute':
        if not args.backend:
            print("Error: Se requiere especificar un backend para ejecutar")
            sys.exit(1)
        execute_circuit(provider, args.backend)

if __name__ == "__main__":
    main()

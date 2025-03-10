#!/usr/bin/env python3
"""
coremind_quantum_cli.py - Interfaz avanzada de línea de comandos para IBM Quantum

Una herramienta de CLI sofisticada para interactuar con IBM Quantum Experience
dentro del marco de trabajo CoreMind powered with AI.

Uso:
    python coremind_quantum_cli.py --token TU_TOKEN [--action {status,execute,list,jobs,results,custom}]
                                   [--backend NOMBRE_BACKEND] [--job-id ID_TRABAJO]
                                   [--circuit-type {bell,ghz,qft,custom}] [--qubits NUM_QUBITS]
                                   [--shots NUM_SHOTS] [--noise-model {true,false}]
                                   [--optimization-level {0,1,2,3}] [--output {text,json,csv}]
                                   [--save-path RUTA_GUARDADO] [--verbose]
"""

import argparse
import json
import time
import csv
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Importaciones de Qiskit
from qiskit import IBMQ, Aer, QuantumCircuit, execute, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_city
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import IBMQBackend, IBMQJob
from qiskit.providers.ibmq.job import job_status_message
from qiskit.providers.ibmq.job.exceptions import IBMQJobApiError
from qiskit.providers.ibmq.exceptions import IBMQProviderError, IBMQAccountError
from qiskit.quantum_info import state_fidelity
from qiskit.result import Result
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit.library import (
    QFT, GroverOperator, PhaseEstimation, 
    EfficientSU2, RealAmplitudes, ZZFeatureMap
)


# Configuración del sistema de logging
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("coremind_quantum")


class CoreMindQuantumManager:
    """Gestor principal para interactuar con IBM Quantum Experience."""
    
    def __init__(self, token: str, verbose: bool = False):
        """
        Inicializa el gestor de quantum computing.
        
        Args:
            token: Token de API para IBM Quantum Experience
            verbose: Si es True, activa logging detallado
        """
        self.token = token
        self.provider = None
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self._setup_connection()
    
    def _setup_connection(self) -> None:
        """Establece la conexión con IBM Quantum."""
        try:
            logger.info("Configurando conexión con IBM Quantum...")
            
            # Comprobar si ya estamos autenticados
            if IBMQ.active_account():
                IBMQ.disable_account()
                
            IBMQ.save_account(self.token, overwrite=True)
            IBMQ.load_account()
            self.provider = IBMQ.get_provider()
            
            if self.verbose:
                logger.debug(f"Conexión establecida correctamente con el proveedor: {self.provider.name()}")
                
            logger.info("✓ Conexión establecida con IBM Quantum")
        except IBMQAccountError as e:
            logger.error(f"Error de autenticación: {str(e)}")
            sys.exit(1)
        except IBMQProviderError as e:
            logger.error(f"Error del proveedor: {str(e)}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error inesperado al configurar la API: {str(e)}")
            sys.exit(1)

    def get_backends(self, operational_only: bool = False, simulator_only: bool = False,
                    min_qubits: Optional[int] = None) -> List[IBMQBackend]:
        """
        Obtiene los backends disponibles según criterios de filtrado.
        
        Args:
            operational_only: Si es True, filtra solo backends operativos
            simulator_only: Si es True, filtra solo simuladores
            min_qubits: Número mínimo de qubits requeridos
            
        Returns:
            Lista de backends filtrados
        """
        try:
            filters = {}
            
            if operational_only:
                filters['operational'] = True
            if simulator_only:
                filters['simulator'] = True
            if min_qubits:
                filters['min_qubits'] = min_qubits
                
            backends = self.provider.backends(**filters)
            return backends
        except Exception as e:
            logger.error(f"Error al obtener backends: {str(e)}")
            return []

    def list_backends(self, output_format: str = 'text', save_path: Optional[str] = None) -> None:
        """
        Lista todos los backends disponibles con información detallada.
        
        Args:
            output_format: Formato de salida ('text', 'json', o 'csv')
            save_path: Ruta donde guardar la salida
        """
        try:
            backends = self.get_backends()
            
            if not backends:
                logger.warning("No se encontraron backends disponibles")
                return
                
            backend_data = []
            for backend in backends:
                try:
                    status = backend.status()
                    config = backend.configuration()
                    
                    # Extraer información relevante
                    backend_info = {
                        "nombre": backend.name(),
                        "tipo": "Simulador" if config.simulator else "Dispositivo cuántico",
                        "qubits": config.n_qubits,
                        "operativo": status.operational,
                        "disponible": True if status.status_msg == 'active' else False,
                        "trabajos_pendientes": status.pending_jobs,
                        "max_shots": config.max_shots,
                        "memoria": hasattr(config, 'memory') and config.memory,
                        "basis_gates": ', '.join(config.basis_gates) if hasattr(config, 'basis_gates') else "N/A"
                    }
                    
                    if not config.simulator:
                        try:
                            properties = backend.properties()
                            # Calcular tiempos promedio de coherencia T1 y T2
                            if properties and hasattr(properties, 'qubits'):
                                t1_values = [qubit[1].value for qubit in enumerate(properties.qubits) 
                                           for item in qubit[1] if item.name == 'T1']
                                t2_values = [qubit[1].value for qubit in enumerate(properties.qubits) 
                                           for item in qubit[1] if item.name == 'T2']
                                
                                if t1_values:
                                    backend_info["t1_promedio"] = f"{sum(t1_values)/len(t1_values):.2f} μs"
                                if t2_values:
                                    backend_info["t2_promedio"] = f"{sum(t2_values)/len(t2_values):.2f} μs"
                        except Exception as e:
                            if self.verbose:
                                logger.debug(f"No se pudieron obtener propiedades para {backend.name()}: {str(e)}")
                    
                    backend_data.append(backend_info)
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"Error al procesar el backend {backend.name()}: {str(e)}")
            
            # Ordenar por número de qubits (descendente)
            backend_data = sorted(backend_data, key=lambda x: x['qubits'], reverse=True)
            
            # Generar salida en el formato solicitado
            if output_format == 'json':
                output = json.dumps(backend_data, indent=2, ensure_ascii=False)
                if save_path:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(output)
                else:
                    print(output)
            elif output_format == 'csv':
                if save_path:
                    with open(save_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=backend_data[0].keys())
                        writer.writeheader()
                        writer.writerows(backend_data)
                else:
                    # Crear una salida de texto CSV en la consola
                    output = ','.join(backend_data[0].keys()) + '\n'
                    for row in backend_data:
                        output += ','.join(str(value) for value in row.values()) + '\n'
                    print(output)
            else:  # text (default)
                # Crear tablas separadas para dispositivos reales y simuladores
                real_devices = [b for b in backend_data if b['tipo'] == "Dispositivo cuántico"]
                simulators = [b for b in backend_data if b['tipo'] == "Simulador"]
                
                print("\n=== DISPOSITIVOS CUÁNTICOS ===")
                if real_devices:
                    headers = ["Nombre", "Qubits", "Estado", "Cola", "T1 Promedio", "Basis Gates"]
                    table_data = [
                        [
                            d['nombre'], 
                            d['qubits'],
                            "✓ Operativo" if d['operativo'] else "✗ No operativo",
                            d['trabajos_pendientes'],
                            d.get('t1_promedio', "N/A"),
                            d['basis_gates'][:50] + "..." if len(d['basis_gates']) > 50 else d['basis_gates']
                        ] 
                        for d in real_devices
                    ]
                    print(tabulate(table_data, headers=headers, tablefmt="grid"))
                else:
                    print("No hay dispositivos cuánticos disponibles")
                
                print("\n=== SIMULADORES ===")
                if simulators:
                    headers = ["Nombre", "Qubits", "Estado", "Max Shots", "Memoria"]
                    table_data = [
                        [
                            d['nombre'], 
                            d['qubits'],
                            "✓ Operativo" if d['operativo'] else "✗ No operativo",
                            d['max_shots'],
                            "Sí" if d['memoria'] else "No"
                        ] 
                        for d in simulators
                    ]
                    print(tabulate(table_data, headers=headers, tablefmt="grid"))
                else:
                    print("No hay simuladores disponibles")
                
                # Guardar en archivo si se especifica ruta
                if save_path:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write("=== DISPOSITIVOS CUÁNTICOS ===\n")
                        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
                        f.write("\n\n=== SIMULADORES ===\n")
                        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
                    print(f"\nResultados guardados en: {save_path}")
                
        except Exception as e:
            logger.error(f"Error al listar backends: {str(e)}")

    def check_backend_status(self, backend_name: str, output_format: str = 'text',
                           save_path: Optional[str] = None) -> None:
        """
        Verifica el estado detallado de un backend específico.
        
        Args:
            backend_name: Nombre del backend a verificar
            output_format: Formato de salida ('text', 'json', o 'csv')
            save_path: Ruta donde guardar la salida
        """
        try:
            backend = self.provider.get_backend(backend_name)
            status = backend.status()
            config = backend.configuration()
            
            # Recopilar información básica
            status_info = {
                "nombre": backend.name(),
                "tipo": "Simulador" if config.simulator else "Dispositivo cuántico",
                "qubits": config.n_qubits,
                "operativo": status.operational,
                "estado": status.status_msg,
                "trabajos_pendientes": status.pending_jobs,
                "max_shots": config.max_shots,
                "max_experiments": config.max_experiments
            }
            
            # Agregar propiedades detalladas para dispositivos reales
            if not config.simulator:
                try:
                    properties = backend.properties()
                    
                    # Información de coherencia por qubit
                    qubit_props = {}
                    for qubit_idx, qubit in enumerate(properties.qubits):
                        qubit_data = {}
                        for item in qubit:
                            if item.name in ['T1', 'T2', 'frequency', 'readout_error']:
                                if item.name in ['T1', 'T2']:
                                    qubit_data[item.name] = f"{item.value:.2f} μs"
                                elif item.name == 'frequency':
                                    qubit_data[item.name] = f"{item.value / 1e9:.5f} GHz"
                                else:
                                    qubit_data[item.name] = f"{item.value:.5f}"
                        qubit_props[f"qubit_{qubit_idx}"] = qubit_data
                    
                    # Calcular promedios
                    t1_values = [qubit[1].value for qubit in enumerate(properties.qubits) 
                               for item in qubit[1] if item.name == 'T1']
                    t2_values = [qubit[1].value for qubit in enumerate(properties.qubits) 
                               for item in qubit[1] if item.name == 'T2']
                    readout_errors = [qubit[1].value for qubit in enumerate(properties.qubits) 
                                   for item in qubit[1] if item.name == 'readout_error']
                    
                    # Añadir promedios al estado
                    if t1_values:
                        status_info["t1_promedio"] = f"{sum(t1_values)/len(t1_values):.2f} μs"
                        status_info["t1_min"] = f"{min(t1_values):.2f} μs"
                        status_info["t1_max"] = f"{max(t1_values):.2f} μs"
                    
                    if t2_values:
                        status_info["t2_promedio"] = f"{sum(t2_values)/len(t2_values):.2f} μs"
                        status_info["t2_min"] = f"{min(t2_values):.2f} μs"
                        status_info["t2_max"] = f"{max(t2_values):.2f} μs"
                    
                    if readout_errors:
                        status_info["error_lectura_promedio"] = f"{sum(readout_errors)/len(readout_errors):.5f}"
                    
                    # Información sobre puertas
                    status_info["basis_gates"] = config.basis_gates
                    
                    # Añadir propiedades por qubit
                    status_info["propiedades_qubits"] = qubit_props
                    
                    # Información sobre conectividad
                    if hasattr(config, 'coupling_map') and config.coupling_map:
                        status_info["mapa_acoplamiento"] = config.coupling_map
                
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"Error al obtener propiedades detalladas: {str(e)}")
                    status_info["propiedades_detalladas"] = "No disponibles"
            
            # Generar salida en el formato solicitado
            if output_format == 'json':
                output = json.dumps(status_info, indent=2, ensure_ascii=False)
                if save_path:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(output)
                else:
                    print(output)
            elif output_format == 'csv':
                # Aplanar el diccionario para CSV
                flat_dict = self._flatten_dict(status_info)
                
                if save_path:
                    with open(save_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        for key, value in flat_dict.items():
                            writer.writerow([key, value])
                else:
                    for key, value in flat_dict.items():
                        print(f"{key},{value}")
            else:  # text (default)
                print(f"\n=== ESTADO DEL BACKEND: {backend_name} ===\n")
                
                print(f"Tipo: {status_info['tipo']}")
                print(f"Número de qubits: {status_info['qubits']}")
                print(f"Estado: {'✓ Operativo' if status_info['operativo'] else '✗ No operativo'} ({status_info['estado']})")
                print(f"Trabajos en cola: {status_info['trabajos_pendientes']}")
                print(f"Máximo de shots: {status_info['max_shots']}")
                print(f"Máximo de experimentos: {status_info['max_experiments']}")
                
                if not config.simulator:
                    print("\n--- Métricas de rendimiento ---")
                    if 't1_promedio' in status_info:
                        print(f"T1 promedio: {status_info['t1_promedio']} (min: {status_info['t1_min']}, max: {status_info['t1_max']})")
                    if 't2_promedio' in status_info:
                        print(f"T2 promedio: {status_info['t2_promedio']} (min: {status_info['t2_min']}, max: {status_info['t2_max']})")
                    if 'error_lectura_promedio' in status_info:
                        print(f"Error de lectura promedio: {status_info['error_lectura_promedio']}")
                    
                    print("\n--- Puertas cuánticas soportadas ---")
                    gates_list = status_info.get('basis_gates', [])
                    # Mostrar las puertas en múltiples líneas si hay muchas
                    line_length = 0
                    gate_lines = []
                    current_line = []
                    
                    for gate in gates_list:
                        if line_length + len(gate) + 2 > 80:  # +2 por la coma y el espacio
                            gate_lines.append(', '.join(current_line))
                            current_line = [gate]
                            line_length = len(gate)
                        else:
                            current_line.append(gate)
                            line_length += len(gate) + 2
                    
                    if current_line:
                        gate_lines.append(', '.join(current_line))
                    
                    for line in gate_lines:
                        print(f"  {line}")
                    
                    # Mostrar información por qubit en una tabla
                    if 'propiedades_qubits' in status_info:
                        print("\n--- Propiedades por qubit ---")
                        qubit_data = []
                        headers = ["Qubit", "T1 (μs)", "T2 (μs)", "Frecuencia (GHz)", "Error Lectura"]
                        
                        for qubit_id, props in status_info['propiedades_qubits'].items():
                            qubit_idx = int(qubit_id.split('_')[1])
                            row = [
                                qubit_idx,
                                props.get('T1', 'N/A'),
                        
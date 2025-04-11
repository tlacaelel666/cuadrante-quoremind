#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coremind_quantum_cli.py - Interfaz avanzada de línea de comandos para IBM Quantum

Una herramienta de CLI sofisticada para interactuar con IBM Quantum Experience
dentro del marco de trabajo CoreMind powered with AI.
Fecha: 07-abril-2025.
Autor: Jacobo Tlacaelel Mina Rodríguez, optimizado y documentación Gemini AI, Cloude anthropic.
version: QuoreMind v1.1.1

Uso:
    python coremind_quantum_cli.py --token TU_TOKEN [--action {status,execute,list,jobs,results,custom}]
                                   [--backend NOMBRE_BACKEND] [--job-id ID_TRABAJO]
                                   [--circuit-type {bell,ghz,qft,grover,vqe,su2,zz,qv,custom}] [--circuit-file RUTA_QASM]
                                   [--qubits NUM_QUBITS] [--shots NUM_SHOTS] [--noise-model {true,false}]
                                   [--optimization-level {0,1,2,3}] [--output {text,json,csv,plot}]
                                   [--save-path RUTA_GUARDADO] [--plot-results] [--verbose] [--timeout TIMEOUT]
"""

import argparse
import json
import time
import csv
import os
import sys
import logging
import tempfile
import webbrowser
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
try:
    from tabulate import tabulate  # Necesita: pip install tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Advertencia: 'tabulate' no instalado. Salida de texto será menos formateada. Instalar con: pip install tabulate")

# Importaciones de Qiskit
from qiskit import QuantumCircuit, transpile, execute
from qiskit.visualization import (
    plot_histogram, plot_bloch_multivector, plot_state_city,
    plot_gate_map, plot_error_map, plot_circuit_layout
)
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_provider import IBMProvider, IBMJob, JobStatus, IBMJobApiError, IBMProviderError, IBMAccountError
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit_aer.noise import NoiseModel
from qiskit.circuit.library import (
    QFT, GroverOperator, EfficientSU2, ZZFeatureMap, QuantumVolume
)

# Configuración del sistema de logging
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("coremind_quantum.log", mode='a', encoding='utf-8')  # Añadir encoding
    ]
)
logger = logging.getLogger("coremind quantum")


class CoreMindQuantumManager:
    """Gestor principal para interactuar con IBM Quantum Experience."""

    def __init__(self, token: str, verbose: bool = False, timeout: int = 300):
        """
        Inicializa el gestor.
        Args:
            token: Token de API.
            verbose: Activa logging detallado.
            timeout: Tiempo máximo de espera para operaciones de red (ej. retrieve_job).
        """
        self.token = token
        self.provider = IBMProvider(token=token)
        self.verbose = verbose
        self.timeout = timeout  # Timeout para operaciones bloqueantes (ej. job results)
        self.session_start_time = datetime.now()

        # Cache simple para propiedades (refrescar cada día) y resultados
        self._backend_properties_cache: Dict[str, Any] = {}
        self._results_cache: Dict[str, Result] = {}  # Cache de resultados por job_id

        if verbose:
            logger.setLevel(logging.DEBUG)

        self._display_banner()

    def _display_banner(self) -> None:
        """Muestra un banner de inicio."""
        banner = """
        ┌──────────────────────────────────────────────────────┐
        │                                                      │
        │     ██████╗ ██████╗ ██████╗ ███████╗███╗   ███╗     │
        │    ██╔════╝██╔═══██╗██╔══██╗██╔════╝████╗ ████║     │
        │    ██║     ██║   ██║██████╔╝█████╗  ██╔████╔██║     │
        │    ██║     ██║   ██║██╔══██╗██╔══╝  ██║╚██╔╝██║     │
        │    ╚██████╗╚██████╔╝██║  ██║███████╗██║ ╚═╝ ██║     │
        │     ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝     │
        │                                                      │
        │  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗███╗   ███╗  │
        │ ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║████╗ ████║  │
        │ ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║  │
        │ ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║  │
        │ ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║  │
        │  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝  │
        │                                                      │
        │              CLI IBM Quantum Experience              │
        │                     v2.0.1 (Completado)              │
        │                                                      │
        └──────────────────────────────────────────────────────┘
        """
        print(banner)
        print(f"Sesión iniciada: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

    def list_backends(self, output_format: str = 'text', save_path: Optional[str] = None) -> None:
        """Lista backends disponibles."""
        try:
            logger.info("Obteniendo lista de backends...")
            backends = self.provider.backends()
            if not backends:
                logger.warning("No se encontraron backends.")
                self._format_output([], output_format=output_format, save_path=save_path, title="Backends Disponibles")
                return

            backend_data = []
            # Usar ThreadPoolExecutor para obtener propiedades en paralelo (puede acelerar)
            futures = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                for backend in backends:
                    futures[executor.submit(self._get_backend_info, backend)] = backend.name()

                for future in as_completed(futures):
                    backend_name = futures[future]
                    try:
                        info = future.result()
                        if info:
                            backend_data.append(info)
                    except Exception as e_future:
                        logger.warning(f"Error obteniendo info para backend {backend_name}: {e_future}")

            backend_data = sorted(backend_data, key=lambda x: (x.get('Tipo') == 'Dispositivo Cuántico', x.get('Qubits', 0)), reverse=True)

            if output_format == 'plot':
                self._create_backend_visualization(backend_data, save_path)
            else:
                headers = ["Nombre", "Tipo", "Qubits", "Operativo", "Estado", "Cola", "Max Shots", "T1 μs (Avg)", "T2 μs (Avg)"]
                present_headers = [h for h in headers if any(h in d for d in backend_data)]
                self._format_output(backend_data, headers=present_headers, output_format=output_format, save_path=save_path, title="Backends Disponibles")

        except Exception as e:
            logger.error(f"Error al listar backends: {e}")
            if self.verbose: import traceback; logger.debug(traceback.format_exc())

    def _get_backend_info(self, backend) -> Optional[Dict]:
        """Helper para obtener info de un backend (para ThreadPool)."""
        try:
            status = backend.status()
            config = backend.configuration()
            backend_info = {
                "Nombre": backend.name(),
                "Tipo": "Simulador" if config.simulator else "Dispositivo Cuántico",
                "Qubits": config.n_qubits,
                "Operativo": "✓" if status.operational else "✗",
                "Estado": status.status_msg,
                "Cola": status.pending_jobs,
                "Max Shots": getattr(config, 'max_shots', 'N/A'),
                "Memoria": getattr(config, 'memory', False)
            }
            if not config.simulator:
                props = self._get_cached_properties(backend)
                if props:
                    t1s = [props.t1(q) * 1e6 for q in range(config.n_qubits) if props.t1(q) is not None]
                    t2s = [props.t2(q) * 1e6 for q in range(config.n_qubits) if props.t2(q) is not None]
                    if t1s: backend_info["T1 μs (Avg)"] = f"{np.mean(t1s):.1f}"
                    if t2s: backend_info["T2 μs (Avg)"] = f"{np.mean(t2s):.1f}"
                backend_info["Basis Gates"] = getattr(config, 'basis_gates', 'N/A')
            return backend_info
        except Exception as e:
            logger.debug(f"Error interno obteniendo info para {backend.name()}: {e}")
            return None  # Devolver None si falla

    def _get_cached_properties(self, backend) -> Optional[Any]:
        """Obtiene propiedades de backend desde cache o API."""
        if backend.configuration().simulator: return None
        cache_key = f"{backend.name()}_{datetime.now().strftime('%Y%m%d')}"  # Cache diario
        if cache_key not in self._backend_properties_cache:
            try:
                logger.debug(f"Consultando propiedades de backend para {backend.name()}...")
                self._backend_properties_cache[cache_key] = backend.properties()
            except Exception as e:
                logger.warning(f"No se pudieron obtener propiedades para {backend.name()}: {e}")
                self._backend_properties_cache[cache_key] = None  # Guardar None para no reintentar hoy
        return self._backend_properties_cache[cache_key]

    def check_backend_status(self, backend_name: str, output_format: str = 'text',
                             save_path: Optional[str] = None) -> None:
        """Verifica el estado detallado de un backend."""
        try:
            logger.info(f"Obteniendo estado detallado para backend: {backend_name}...")
            backend = self.provider.get_backend(backend_name)
            status = backend.status()
            config = backend.configuration()

            status_info = {
                "nombre": backend.name(),
                "tipo": "Simulador" if config.simulator else "Dispositivo Cuántico",
                "qubits": config.n_qubits,
                "operativo": status.operational,
                "estado_msg": status.status_msg,
                "cola_trabajos": status.pending_jobs,
                "version": getattr(backend, 'backend_version', 'N/A'),
                "max_shots": getattr(config, 'max_shots', 'N/A'),
                "max_experiments": getattr(config, 'max_experiments', 'N/A'),
                "memoria_clasica": getattr(config, 'memory', False),
                "puertas_base": getattr(config, 'basis_gates', []),
                "mapa_acoplamiento": getattr(config, 'coupling_map', None)
            }

            qubit_details_table_str = ""
            if not config.simulator:
                props = self._get_cached_properties(backend)
                if props:
                    qubit_details = []
                    t1s, t2s, freqs, read_errs = [], [], [], []
                    headers_q = ["Qubit", "T1 (μs)", "T2 (μs)", "Frec (GHz)", "Error Lec."]
                    for q in range(config.n_qubits):
                        t1 = props.t1(q) * 1e6 if props.t1(q) is not None else None
                        t2 = props.t2(q) * 1e6 if props.t2(q) is not None else None
                        fr = props.frequency(q) / 1e9 if props.frequency(q) is not None else None
                        readout_error_val = None
                        if hasattr(props, 'readout_error'): readout_error_val = props.readout_error(q)
                        elif hasattr(props, 'readout_errors'): readout_error_val = props.readout_errors(q)

                        re = readout_error_val if readout_error_val is not None else None
                        qubit_details.append([
                            q,
                            f"{t1:.1f}" if t1 is not None else "N/A",
                            f"{t2:.1f}" if t2 is not None else "N/A",
                            f"{fr:.5f}" if fr is not None else "N/A",
                            f"{re:.5f}" if re is not None else "N/A"
                        ])
                        if t1: t1s.append(t1)
                        if t2: t2s.append(t2)
                        if fr: freqs.append(fr)
                        if re: read_errs.append(re)

                    if HAS_TABULATE:
                        qubit_details_table_str = tabulate(qubit_details, headers=headers_q, tablefmt="grid")
                    else:  # Fallback
                        qubit_details_table_str = "\n".join([" | ".join(map(str, row)) for row in [headers_q] + qubit_details])

                    if t1s: status_info["t1_avg_us"] = f"{np.mean(t1s):.1f}"
                    if t2s: status_info["t2_avg_us"] = f"{np.mean(t2s):.1f}"
                    if read_errs: status_info["readout_err_avg"] = f"{np.mean(read_errs):.5f}"
                    status_info["propiedades_por_qubit"] = {f"Q{i}": dict(zip(headers_q[1:], row[1:])) for i, row in enumerate(qubit_details)}

                else:
                    status_info["propiedades_qubits"] = "No disponibles o vacías."
            else:
                status_info.pop('mapa_acoplamiento', None)  # No aplica a simuladores

            if status_info.get('mapa_acoplamiento'):
                cmap_list = list(status_info['mapa_acoplamiento'])
                status_info['mapa_acoplamiento_str'] = str(cmap_list)  # Para texto
                status_info['mapa_acoplamiento'] = cmap_list  # Para JSON
            else:
                status_info['mapa_acoplamiento_str'] = "N/A"

            title = f"Estado Detallado de Backend: {backend_name}"
            self._format_output(status_info, output_format=output_format, save_path=save_path, title=title)

            if output_format == 'text':
                if qubit_details_table_str:
                    print(f"\n--- Propiedades por Qubit ---\n{qubit_details_table_str}")
                if status_info.get('mapa_acoplamiento'):
                    print(f"\n--- Mapa de Acoplamiento ---\n{status_info['mapa_acoplamiento_str']}")

        except Exception as e:
            logger.error(f"Error al verificar estado de {backend_name}: {str(e)}")
            if self.verbose: import traceback; logger.debug(traceback.format_exc())

    def _build_circuit(self, circuit_type: str, num_qubits: int, circuit_file: Optional[str] = None) -> QuantumCircuit:
        """Construye un circuito cuántico según el tipo."""
        logger.debug(f"Construyendo circuito tipo '{circuit_type}' para {num_qubits} qubits...")

        if circuit_type not in ['custom', 'vqe_ansatz'] and (num_qubits is None or num_qubits <= 0):
            raise ValueError(f"Se requiere --qubits > 0 para el circuito '{circuit_type}'.")
        if circuit_type in ['bell', 'ghz', 'grover', 'phase_est'] and num_qubits < 2:
            raise ValueError(f"Circuito '{circuit_type}' requiere al menos 2 qubits.")

        qc = None
        if circuit_type == 'bell':
            qc = QuantumCircuit(2, 2, name="Bell")
            qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
        elif circuit_type == 'ghz':
            qc = QuantumCircuit(num_qubits, num_qubits, name=f"GHZ_{num_qubits}")
            qc.h(0)
            for i in range(num_qubits - 1): qc.cx(0, i + 1)
            qc.measure(range(num_qubits), range(num_qubits))
        elif circuit_type == 'qft':
            qc = QuantumCircuit(num_qubits, num_qubits, name=f"QFT_{num_qubits}")
            qc.h(range(num_qubits))  # Superposición inicial
            qc.append(QFT(num_qubits, inverse=False, do_swaps=True), range(num_qubits))
            qc.measure(range(num_qubits), range(num_qubits))
        elif circuit_type == 'grover':
            target_state_str = '1' * num_qubits
            oracle = QuantumCircuit(num_qubits, name='Oracle')
            oracle.h(num_qubits - 1)
            oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)  # Toffoli multi-controlado
            oracle.h(num_qubits - 1)

            grover_op = GroverOperator(oracle, insert_barriers=True)
            iterations = GroverOperator.optimal_num_iterations(num_qubits=num_qubits)
            logger.info(f"Construyendo Grover para buscar |{target_state_str}⟩ con {iterations} iteraciones.")
            qc = QuantumCircuit(num_qubits, num_qubits, name=f"Grover_{num_qubits}")
            qc.h(range(num_qubits))
            qc.append(grover_op.power(iterations), range(num_qubits - 1))
 
qc.measure(range(num_qubits), range(num_qubits))
        elif circuit_type == 'vqe':
            logger.info("Generando ansatz EfficientSU2 para VQE (ejecución completa no soportada en esta acción).")
            qc = EfficientSU2(num_qubits=num_qubits, reps=2, entanglement='linear').decompose()
            qc.name = f"VQE_Ansatz_{num_qubits}"
        elif circuit_type in ['su2', 'zz', 'qv']:
            if circuit_type == 'su2':
                qc_lib = EfficientSU2(num_qubits, reps=3, entanglement='linear')
            elif circuit_type == 'zz':
                qc_lib = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
            else:
                qc_lib = QuantumVolume(num_qubits, seed=int(time.time()))  # Seed aleatorio
            qc = qc_lib.decompose()
            qc.measure_all()
            qc.name = f"{circuit_type.upper()}_{num_qubits}"
        elif circuit_type == 'custom':
            if not circuit_file or not os.path.exists(circuit_file):
                raise ValueError("Para 'custom', se necesita --circuit-file con ruta válida.")
            try:
                logger.info(f"Cargando circuito custom desde {circuit_file}...")
                qc = QuantumCircuit.from_qasm_file(circuit_file)
                if not qc.clbits:
                    logger.warning("Circuito custom no tiene registro clásico. Añadiendo measure_all().")
                    qc.measure_all()
                qc.name = f"Custom_{Path(circuit_file).stem}"
            except Exception as e:
                raise ValueError(f"Error al cargar circuito desde {circuit_file}: {e}")
        else:
            raise ValueError(f"Tipo de circuito '{circuit_type}' no reconocido.")

        logger.info(f"Circuito '{qc.name}' de {qc.num_qubits} qubits construido (Profundidad: {qc.depth()}, Ops: {qc.count_ops()}).")
        return qc

    def execute_circuit(self, backend_name: str, circuit_type: str, num_qubits: int, shots: int,
                        add_noise: bool = False, optimization_level: int = 1,
                        circuit_file: Optional[str] = None) -> Optional[Tuple[IBMJob, Result]]:
        """Ejecuta un circuito cuántico."""
        try:
            logger.info(f"Preparando ejecución: Circuito='{circuit_type}', Backend='{backend_name}', Shots={shots}, Ruido={add_noise}, Opt={optimization_level}")
            backend = self.provider.get_backend(backend_name)
            is_simulator = backend.configuration().simulator

            circuit = self._build_circuit(circuit_type, num_qubits, circuit_file)

            noise_model_instance = None
            if add_noise and is_simulator:
                logger.info("Intentando generar modelo de ruido desde backend real...")
                real_backend = None
                try:
                    real_backends = self.provider.backends(simulator=False, operational=True, min_qubits=circuit.num_qubits)
                    if real_backends:
                        real_backend = min(real_backends, key=lambda b: b.status().pending_jobs)
                        logger.info(f"Usando propiedades de '{real_backend.name()}' para modelo de ruido.")
                        properties = self._get_cached_properties(real_backend)
                        if properties:
                            noise_model_instance = NoiseModel.from_backend(properties)
                            logger.info("✓ Modelo de ruido generado desde backend real.")
                        else:
                            logger.warning("No se pudieron obtener propiedades del backend real.")
                    else:
                        logger.warning("No se encontró backend real adecuado para generar modelo de ruido.")
                except Exception as e_noise:
                    logger.warning(f"Error generando modelo de ruido: {e_noise}. Ejecutando sin ruido.")
            elif add_noise and not is_simulator:
                logger.warning("La opción --noise-model solo aplica a simuladores Aer. Ejecutando en hardware real (con su ruido inherente).")

            logger.info(f"Transpilando circuito '{circuit.name}' para '{backend_name}' (opt={optimization_level})...")
            transpiled_circuit = transpile(circuit, backend=backend, optimization_level=optimization_level)
            logger.info(f"Circuito transpilado: Profundidad={transpiled_circuit.depth()}, Operaciones={transpiled_circuit.count_ops()}")

            logger.info(f"Enviando trabajo a '{backend_name}' (shots={shots})...")
            execute_options = {'shots': shots}
            if noise_model_instance and 'aer_simulator' in backend.name():
                execute_options['noise_model'] = noise_model_instance
            if getattr(backend.configuration(), 'memory', False):
                execute_options['memory'] = True

            job = execute(transpiled_circuit, backend, **execute_options)
            job_id = job.job_id()
            logger.info(f"Trabajo enviado con ID: {job_id}")

            logger.info("Monitoreando trabajo (puede tardar)...")
            start_time = time.time()
            while job.status() not in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]:
                current_status = job.status()
                elapsed_time = time.time() - start_time
                logger.info(f"  Estado actual: {current_status.name} (Tiempo transcurrido: {elapsed_time:.1f}s)")
                if elapsed_time > self.timeout:
                    logger.warning(f"Timeout ({self.timeout}s) alcanzado esperando el trabajo {job_id}. Puedes verificar estado/resultados más tarde.")
                    return job, None  # Devolver job pero no resultado
                time.sleep(10)  # Esperar 10s entre chequeos

            final_status = job.status()
            logger.info(f"Trabajo {job_id} finalizado con estado: {final_status.name}")

            if final_status == JobStatus.DONE:
                logger.info("Obteniendo resultados...")
                result = job.result()
                logger.info(f"✓ Ejecución completada exitosamente.")
                return job, result
            else:
                logger.error(f"El trabajo {job_id} falló o fue cancelado. Mensaje: {job.error_message()}")
                return job, None  # Devolver job pero no resultado

        except IBMJobApiError as e:
            logger.error(f"Error API (trabajo): {e}")
        except IBMProviderError as e:
            logger.error(f"Error proveedor (backend '{backend_name}'): {e}")
        except IBMAccountError as e:
            logger.error(f"Error cuenta IBMQ: {e}")
        except ValueError as e:
            logger.error(f"Error en valor/parámetro: {e}")
        except Exception as e:
            logger.error(f"Error inesperado durante ejecución: {type(e).__name__} - {e}")
            if self.verbose: import traceback; logger.debug(traceback.format_exc())

        return None  # Falló la ejecución

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el estado de un trabajo específico."""
        logger.info(f"Consultando estado del trabajo ID: {job_id}")
        try:
            job = self.provider.retrieve_job(job_id)
            status = job.status()
            backend_name = job.backend().name() if job.backend() else "Desconocido"
            info = {
                "job_id": job.job_id(),
                "backend": backend_name,
                "status": status.name if status else "Desconocido",
                "mensaje_estado": job.status_msg if status else "N/A",
                "tiempo_creacion": job.creation_date().isoformat() if job.creation_date() else "N/A",
                "tiempo_por_paso": job.time_per_step() if job.time_per_step() else {}
            }
            if hasattr(job, 'error_message') and job.error_message():
                info["mensaje_error"] = job.error_message()
            logger.info(f"Estado obtenido para {job_id}: {info['status']}")
            return info
        except IBMJobApiError as e:
            logger.error(f"Error API recuperando trabajo {job_id}: {e}")
        except Exception as e:
            logger.error(f"Error inesperado recuperando trabajo {job_id}: {e}")
        return None

    def list_jobs(self, limit: int = 10, backend_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lista trabajos recientes, opcionalmente filtrando por backend."""
        logger.info(f"Listando los últimos {limit} trabajos" + (f" para backend '{backend_name}'." if backend_name else "."))
        try:
            job_list = self.provider.jobs(limit=limit, backend_name=backend_name, descending=True)
            jobs_data = []
            for job in job_list:
                status = job.status()
                data = {
                    "ID Trabajo": job.job_id(),
                    "Backend": job.backend().name() if job.backend() else "N/A",
                    "Estado": status.name if status else "N/A",
                    "Fecha Creación": job.creation_date().isoformat() if job.creation_date() else "N/A",
                    "Tags": job.tags() if hasattr(job, 'tags') else []
                }
                jobs_data.append(data)
            logger.info(f"Encontrados {len(jobs_data)} trabajos.")
            return jobs_data
        except Exception as e:
            logger.error(f"Error al listar trabajos: {e}")
            return []

    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene y formatea resultados de un trabajo completado."""
        if job_id in self._results_cache:
            logger.info(f"Recuperando resultados cacheados para trabajo ID: {job_id}")
            result_obj = self._results_cache[job_id]
            return self._process_result_object(result_obj, job_id)

        logger.info(f"Obteniendo resultados para el trabajo ID: {job_id}")
        try:
            job = self.provider.retrieve_job(job_id)
            status = job.status()

            if status == JobStatus.DONE:
                logger.info("Trabajo completado. Recuperando objeto Result...")
                result = job.result(timeout=self.timeout)
                logger.info("✓ Resultados obtenidos.")
                results_data = self._process_result_object(result, job_id, job.backend().name())
                self._results_cache[job_id] = result  # Guardar objeto Result crudo en cache
                return results_data
            elif status in [JobStatus.ERROR, JobStatus.CANCELLED]:
                logger.error(f"El trabajo {job_id} finalizó con estado {status.name}. Error: {job.error_message()}")
                return {"job_id": job_id, "status": status.name, "error": job.error_message()}
            else:
                logger.warning(f"El trabajo {job_id} aún no ha terminado (Estado: {status.name}).")
                return {"job_id": job_id, "status": status.name, "mensaje": "Trabajo aún no completado."}

        except IBMJobApiError as e:
            logger.error(f"Error API obteniendo resultados de {job_id}: {e}")
        except Exception as e:
            logger.error(f"Error inesperado obteniendo resultados de {job_id}: {e}")
        return None

    def _process_result_object(self, result: Result, job_id: str, backend_name: Optional[str] = None) -> Dict[str, Any]:
        """Procesa un objeto Result de Qiskit a un diccionario formateado."""
        results_data = {
            "job_id": job_id,
            "backend": backend_name or result.backend_name,
            "status": "DONE",  # Asumimos que llegó aquí porque estaba DONE
            "exito": result.success,
            "fecha_ejecucion": result.date.isoformat(),
            "tiempo_ejecucion_backend_s": getattr(result, 'time_taken', None),
            "shots": result.shots,
            "resultados_experimentos": []
        }
        for i, exp_result in enumerate(result.results):
            exp_data = {"experimento": i}
            if hasattr(exp_result, 'header') and hasattr(exp_result.header, 'name'):
                exp_data["nombre_circuito"] = exp_result.header.name
            if hasattr(exp_result.data, 'counts'):
                exp_data["counts"] = dict(exp_result.data.counts)
            if hasattr(exp_result.data, 'statevector'):
                sv = exp_result.data.statevector
                exp_data["statevector"] = [(c.real, c.imag) for c in sv.data]
            if hasattr(exp_result.data, 'memory'):
                mem = exp_result.data.memory
                exp_data["memory_counts"] = len(mem) if mem else 0
                exp_data["memory_preview"] = mem[:min(len(mem), 5)] if mem else []

            results_data["resultados_experimentos"].append(exp_data)
        return results_data

    def plot_results(self, results_data: Dict[str, Any], save_path_prefix: Optional[str] = None) -> None:
        """Genera gráficas a partir de los datos de resultados."""
        if not results_data or not results_data.get("resultados_experimentos"):
            logger.warning("No hay datos de resultados para graficar.")
            return

        logger.info("Generando gráficas de resultados...")
        job_id_short = results_data.get("job_id", "unknown_job")[:8]
        figs_created = []

        for i, exp_result_data in enumerate(results_data["resultados_experimentos"]):
            exp_name = exp_result_data.get("nombre_circuito", f"exp_{i}")
            plot_filename_base = f"{save_path_prefix}_job_{job_id_short}_{exp_name}" if save_path_prefix else None

            # Graficar histograma
            if "counts" in exp_result_data:
                try:
                    fig_hist = plot_histogram(exp_result_data["counts"], title=f"Histograma - {exp_name} ({job_id_short})", figsize=(10, 6))
                    figs_created.append(fig_hist)
                    if plot_filename_base:
                        filename = f"{plot_filename_base}_hist.png"
                        fig_hist.savefig(filename)
                        logger.info(f"Histograma guardado en {filename}")
                    else:
                        plt.show(block=False)  # No bloquear si se muestran varias
                except Exception as e_hist:
                    logger.error(f"Error generando histograma para {exp_name}: {e_hist}")

            # Graficar estado
            if "statevector" in exp_result_data:
                try:
                    sv_tuples = exp_result_data["statevector"]
                    sv_complex = np.array([complex(r, i) for r, i in sv_tuples])
                    state = Statevector(sv_complex)
                    num_qubits = state.num_qubits

                    if num_qubits <= 3:  # Bloch
                        fig_bloch = plot_bloch_multivector(state, title=f"Esfera de Bloch - {exp_name} ({job_id_short})")
                        figs_created.append(fig_bloch)
                        if plot_filename_base:
                            filename = f"{plot_filename_base}_bloch.png"
                            fig_bloch.savefig(filename)
                            logger.info(f"Esfera de Bloch guardada en {filename}")
                        else:
                            plt.show(block=False)
                    else:  # City
                        fig_city = plot_state_city(state, title=f"State City - {exp_name} ({job_id_short})", figsize=(12, 8))
                        figs_created.append(fig_city)
                        if plot_filename_base:
                            filename = f"{plot_filename_base}_city.png"
                            fig_city.savefig(filename)
                            logger.info(f"State City guardado en {filename}")
                        else:
                            plt.show(block=False)

                except Exception as e_state:
                    logger.error(f"Error graficando statevector para {exp_name}: {e_state}")

        # Mostrar todas las figuras al final si no se guardaron
        if not save_path_prefix and figs_created:
            logger.info("Mostrando gráficas generadas...")
            plt.show()  # Muestra todas las figuras no bloqueantes
        # Cerrar figuras para liberar memoria
        for fig in figs_created:
            plt.close(fig)


# --- Función Principal y Manejo de Argumentos ---
def main():
    # --- Configuración de Argparse ---
    parser = argparse.ArgumentParser(
        description="CoreMind Quantum CLI v2.0.1 - Interfaz para IBM Quantum.",
        formatter_class=argparse.RawTextHelpFormatter,  # Para formato en help
        epilog="""Ejemplos:
  Listar backends operativos con >4 qubits en JSON:
    python %(prog)s -t MI_TOKEN -a list --output json --min-qubits 5 --operational
  Ver estado detallado de 'ibm_brisbane':
    python %(prog)s -t MI_TOKEN -a status -b ibm_brisbane
  Ejecutar circuito GHZ de 5 qubits en simulador Aer con 2048 shots:
    python %(prog)s -t MI_TOKEN -a execute -b aer_simulator --circuit-type ghz --qubits 5 --shots 2048
  Ejecutar custom QASM y guardar resultados/plots:
    python %(prog)s -t MI_TOKEN -a execute -b ibm_simulator --circuit-type custom --circuit-file ./mi_circuito.qasm --shots 4096 --output json --save-path ejecucion.json --plot-results
  Obtener resultados (y plots) de un trabajo anterior:
    python %(prog)s -t MI_TOKEN -a results --job-id TU_JOB_ID --plot --save-path resultados_job.txt
  Listar los últimos 5 trabajos en 'ibm_sherbrooke':
    python %(prog)s -t MI_TOKEN -a jobs --backend ibm_sherbrooke --limit 5 --output csv
"""
    )
    # Grupo de Argumentos Requeridos
    req_group = parser.add_argument_group('Argumentos Requeridos')
    req_group.add_argument("-t", "--token", required=True, help="Token de API de IBM Quantum Experience.")
    req_group.add_argument("-a", "--action", required=True, choices=['list', 'status', 'execute', 'jobs', 'results', 'custom'],
                           help="Acción a realizar:\n"
                                "  list    - Lista backends disponibles.\n"
                                "  status  - Muestra estado detallado de un backend o job.\n"
                                "  execute - Ejecuta un circuito cuántico.\n"
                                "  jobs    - Lista trabajos recientes.\n"
                                "  results - Obtiene resultados de un trabajo.\n"
                                "  custom  -
                                "  custom  - (No implementado) Acción personalizada.")

    # Grupo de Argumentos Opcionales Generales
    gen_group = parser.add_argument_group('Argumentos Generales')
    gen_group.add_argument("-b", "--backend", help="Nombre del backend a usar (para status, execute, jobs).")
    gen_group.add_argument("-j", "--job-id", help="ID del trabajo (para status, results).")
    gen_group.add_argument("-o", "--output", default='text', choices=['text', 'json', 'csv', 'plot'],
                           help="Formato de salida ('plot' solo para acción 'list').")
    gen_group.add_argument("-s", "--save-path", help="Ruta al archivo donde guardar la salida (texto/json/csv o prefijo para plots).")
    gen_group.add_argument("--timeout", type=int, default=300, help="Timeout en segundos para operaciones de red (ej. obtener resultados).")
    gen_group.add_argument("-v", "--verbose", action='store_true', help="Activar logging detallado (DEBUG).")

    # Grupo de Argumentos para Ejecución
    exec_group = parser.add_argument_group('Argumentos para Acción \'execute\'')
    exec_group.add_argument("--circuit-type", default='bell',
                            choices=['bell', 'ghz', 'qft', 'grover', 'vqe', 'su2', 'zz', 'qv', 'custom'],
                            help="Tipo de circuito a ejecutar (o 'custom' con --circuit-file).")
    exec_group.add_argument("--circuit-file", help="Ruta al archivo QASM v2.0 para --circuit-type 'custom'.")
    exec_group.add_argument("--qubits", type=int, default=None,  # Default None para inferir de custom o requerir
                            help="Número de qubits para circuitos generados (requerido si no es 'custom').")
    exec_group.add_argument("--shots", type=int, default=1024, help="Número de shots (repeticiones).")
    exec_group.add_argument("--noise-model", type=lambda x: (str(x).lower() == 'true'), default=False,
                            help="Aplicar modelo de ruido básico (si es simulador Aer) [true/false].")
    exec_group.add_argument("--optimization-level", type=int, default=1, choices=[0, 1, 2, 3],
                            help="Nivel de optimización de transpilación Qiskit [0-3].")
    exec_group.add_argument("--plot-results", action='store_true',
                            help="Generar y guardar gráficas para la acción 'execute' y 'results'.")

    # Grupo de Argumentos para list_jobs
    jobs_group = parser.add_argument_group('Argumentos para Acción \'jobs\'')
    jobs_group.add_argument("--limit", type=int, default=10, help="Número máximo de trabajos a listar.")

    # --- Parseo y Validación ---
    args = parser.parse_args()

    # Validaciones de argumentos
    if args.action == 'status':
        if not args.backend and not args.job_id:
            parser.error("--action 'status' requiere --backend o --job-id.")
    elif args.action == 'execute':
        if not args.backend: parser.error("--action 'execute' requiere --backend.")
        if args.circuit_type == 'custom':
            if not args.circuit_file: parser.error("--circuit-type 'custom' requiere --circuit-file.")
        elif args.qubits is None or args.qubits <= 0:
            parser.error(f"--action 'execute' con circuit-type '{args.circuit_type}' requiere --qubits > 0.")
    elif args.action == 'results':
        if not args.job_id: parser.error("--action 'results' requiere --job-id.")
    elif args.action == 'jobs' and args.backend:
        logger.info(f"Filtrando trabajos por backend: {args.backend}")

    if args.output == 'plot' and args.action != 'list':
        logger.warning("--output 'plot' solo está implementado para --action 'list'. Usando 'text' en su lugar.")
        args.output = 'text'

    if args.plot_results and args.action not in ['execute', 'results']:
        logger.warning("--plot-results solo tiene efecto con --action 'execute' o 'results'.")
        args.plot_results = False

    if args.save_path and args.plot_results and args.output != 'plot':
        logger.info(f"--save-path ('{args.save_path}') guarda la salida {args.output}. Las gráficas de --plot-results se guardarán con ese prefijo.")

    # --- Ejecución ---
    try:
        manager = CoreMindQuantumManager(token=args.token, verbose=args.verbose, timeout=args.timeout)

        if args.action == 'list':
            manager.list_backends(output_format=args.output, save_path=args.save_path)  # 'plot' se maneja dentro

        elif args.action == 'status':
            if args.job_id:
                job_status_info = manager.get_job_status(args.job_id)
                if job_status_info:
                    manager._format_output(job_status_info, output_format=args.output, save_path=args.save_path, title=f"Estado Trabajo {args.job_id}")
            elif args.backend:
                manager.check_backend_status(args.backend, output_format=args.output, save_path=args.save_path)

        elif args.action == 'execute':
            exec_output = manager.execute_circuit(
                backend_name=args.backend,
                circuit_type=args.circuit_type,
                num_qubits=args.qubits,
                shots=args.shots,
                add_noise=args.noise_model,
                optimization_level=args.optimization_level,
                circuit_file=args.circuit_file
            )
            if exec_output:
                job, result = exec_output
                job_id = job.job_id()
                if result:  # Si la ejecución terminó y devolvió resultado
                    logger.info(f"Ejecución completada (Job ID: {job_id}). Éxito: {result.success}")
                    results_data = manager._process_result_object(result, job_id, args.backend)  # Usar helper interno
                    # Mostrar/Guardar resultados en formato pedido
                    manager._format_output(results_data, output_format=args.output, save_path=args.save_path, title=f"Resultados Ejecución {job_id}")
                    # Graficar si se pide
                    if args.plot_results:
                        plot_save_prefix = os.path.splitext(args.save_path)[0] if args.save_path else f"job_{job_id}"
                        manager.plot_results(results_data, save_path_prefix=plot_save_prefix)
                else:  # Si hubo timeout o error devuelto por execute_circuit
                    logger.warning(f"La ejecución del trabajo {job_id} no produjo resultados finales dentro del timeout/límite.")
            else:  # Si execute_circuit falló antes de devolver job/result
                logger.error(f"La ejecución en {args.backend} falló.")

        elif args.action == 'jobs':
            jobs_list = manager.list_jobs(limit=args.limit, backend_name=args.backend)
            if jobs_list:
                headers = ["ID Trabajo", "Backend", "Estado", "Fecha Creación", "Tags"]
                manager._format_output(jobs_list, headers=headers, output_format=args.output, save_path=args.save_path, title=f"Últimos {args.limit} Trabajos")

        elif args.action == 'results':
            results_data = manager.get_job_results(args.job_id)
            if results_data and results_data.get('status') == 'DONE':
                manager._format_output(results_data, output_format=args.output, save_path=args.save_path, title=f"Resultados Trabajo {args.job_id}")
                # Graficar si se pide
                if args.plot_results:
                    plot_save_prefix = os.path.splitext(args.save_path)[0] if args.save_path else f"job_{args.job_id}"
                    manager.plot_results(results_data, save_path_prefix=plot_save_prefix)

        elif args.action == 'custom':
            logger.warning("Acción 'custom' no implementada.")
            print("Define tu lógica personalizada aquí.")

    except Exception as e:
        logger.exception(f"Error fatal en la ejecución de la CLI.")
        sys.exit(1)
    finally:
        # Asegurar que los plots se cierren
        plt.close('all')


if __name__ == "__main__":
    main()
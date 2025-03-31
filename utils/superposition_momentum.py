#!/usr/bin/env python3

"""
QuantumOS Knob: Sistema Híbrido Cuántico-Bayesiano con Gestión de Momentum en Superposición

Este componente proporciona:
1. Interfaz para la gestión de estados cuánticos en un SO
2. Monitoreo y manipulación del momentum cuántico
3. Optimización de recursos cuánticos y clásicos
4. Persistencia y serialización de estados cuánticos
5. APIs para integración con otros componentes del SO

Autor: Jacobo Tlacaelel Mina Rodríguez
Colaboradores: Equipo QuantumOS
Versión: 0.9.1-alpha
Fecha: 01/04/2025 (Actualizado)
"""

import numpy as np
# Tensorflow importado, pero no usado activamente en esta clase.
# Podría ser para la parte Bayesiana/ML del sistema general.
import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model, save_model
# from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import scipy.stats as stats # No usado directamente aquí
from scipy.fft import fft, ifft
import pickle
import json
import os
import logging
import time
import threading
import queue # No usado directamente aquí
import uuid
import warnings
import hashlib # No usado directamente aquí
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')  # Configuración para headless

# Dependencias específicas para la parte cuántica
try:
    from qiskit import Aer, execute, QuantumCircuit, transpile # IBMQ obsoleto, usar qiskit_ibm_provider
    from qiskit.quantum_info import Statevector, state_fidelity, entropy, partial_trace
    from qiskit.visualization import plot_bloch_multivector, plot_histogram # plot_bloch_multivector puede requerir estado
    # from qiskit.providers.aer.noise import NoiseModel # Usar qiskit_aer.noise
    # from qiskit.ignis.mitigation.measurement import CompleteMeasFitter # Ignis está obsoleto, usar qiskit-experiments
    HAS_QISKIT = True
except ImportError:
    warnings.warn("Qiskit no está instalado. Utilizando modo de simulación limitado.")
    HAS_QISKIT = False
    # Definir clases dummy si Qiskit no está presente para evitar errores
    class DummyQuantumCircuit:
        def __init__(self, *args, **kwargs): pass
        def initialize(self, *args, **kwargs): pass
        def x(self, *args, **kwargs): pass
        def y(self, *args, **kwargs): pass
        def z(self, *args, **kwargs): pass
        def h(self, *args, **kwargs): pass
        def s(self, *args, **kwargs): pass
        def t(self, *args, **kwargs): pass
        def cx(self, *args, **kwargs): pass
        def cz(self, *args, **kwargs): pass
        def qft(self, *args, **kwargs): pass

    class DummyAer:
        @staticmethod
        def get_backend(name):
            if name == 'statevector_simulator':
                # Devolver un objeto simulado si es necesario
                return None # O una clase dummy que simule la ejecución
            raise ImportError("Simulador Qiskit Aer no disponible.")
            
    QuantumCircuit = DummyQuantumCircuit
    Aer = DummyAer
    Statevector = np.ndarray # Usar numpy array como placeholder

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(module)s:%(lineno)d] [%(thread)d] - %(message)s',
    handlers=[
        logging.FileHandler("quantumos_knob.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumOS_Knob")

# Definición de constantes del sistema
PERSISTENCE_DIR = os.environ.get("QUANTUM_OS_DATA", os.path.join(os.path.expanduser("~"), ".quantumos"))
STATE_DIR = os.path.join(PERSISTENCE_DIR, "states")
VIS_DIR = os.path.join(PERSISTENCE_DIR, "visualizations")
CHECKPOINT_DIR = os.path.join(PERSISTENCE_DIR, "checkpoints")
API_VERSION = "v1"
DEFAULT_QUBIT_COUNT = 5
MAX_QUBIT_COUNT = 16 # Reducido a un límite más realista para simulación local
THREAD_POOL_SIZE = 4
CHECKPOINT_INTERVAL = 600  # segundos (10 minutos)

# Creación de directorios necesarios
os.makedirs(PERSISTENCE_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# os.makedirs(os.path.join(PERSISTENCE_DIR, "models"), exist_ok=True) # Relacionado con TF?
# os.makedirs(os.path.join(PERSISTENCE_DIR, "logs"), exist_ok=True) # Usar logging centralizado

# Enumeraciones para el sistema
class ResourcePriority(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

class ExecutionBackend(Enum):
    SIMULATION = auto()
    LOCAL_HARDWARE = auto() # Requiere integración específica
    REMOTE_IBM = auto()     # Requiere qiskit_ibm_provider
    REMOTE_AZURE = auto()   # Requiere azure-quantum
    REMOTE_AMAZON = auto()  # Requiere amazon-braket-sdk
    CUSTOM = auto()

class StateType(Enum):
    PURE = auto()
    MIXED = auto() # Necesitaría matriz de densidad
    ENTANGLED = auto()
    SUPERPOSITION = auto()
    # COHERENT = auto() # Generalmente se aplica a modos de campo, no a qubits

# Clases de excepción personalizadas
class QuantumOSError(Exception): pass
class ResourceUnavailableError(QuantumOSError): pass
class StateDecoherenceError(QuantumOSError): pass # Requeriría simulación de decoherencia
class InvalidStateError(QuantumOSError): pass
class BackendCommunicationError(QuantumOSError): pass
class PersistenceError(QuantumOSError): pass

# Dataclasses para estructuras de datos
@dataclass
class QuantumStateMetadata:
    id: str
    creation_timestamp: float
    last_modified: float
    num_qubits: int
    state_type: StateType
    is_persistent: bool = False
    owner_process: Optional[int] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    fidelity_history: List[Tuple[float, float]] = field(default_factory=list) # (timestamp, fidelity)
    entanglement_metrics: Dict[str, float] = field(default_factory=dict)
    last_checkpoint_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['state_type'] = self.state_type.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumStateMetadata':
        data['state_type'] = StateType[data['state_type']]
        # Asegurar compatibilidad si faltan campos en archivos antiguos
        obj = cls(id=data['id'],
                  creation_timestamp=data.get('creation_timestamp', time.time()),
                  last_modified=data.get('last_modified', time.time()),
                  num_qubits=data['num_qubits'],
                  state_type=data['state_type'])
        # Poblar campos opcionales/nuevos
        for key, value in data.items():
            if hasattr(obj, key) and key not in ['id', 'creation_timestamp', 'last_modified', 'num_qubits', 'state_type']:
                setattr(obj, key, value)
        return obj

@dataclass
class SystemResources: # Placeholder - Necesita implementación real de monitoreo
    available_qubits: int
    quantum_memory_usage: float
    classical_memory_usage: float
    active_processes: int
    backend_status: Dict[str, bool]
    error_rates: Dict[str, float]
    decoherence_times: Dict[str, float] # Depende del backend/hardware
    entanglement_capacity: int # Límite práctico de entrelazamiento

@dataclass
class JobMetadata: # Placeholder - Necesita sistema de gestión de trabajos
    job_id: str
    state_id: str
    submission_time: float
    status: str # (PENDING, RUNNING, COMPLETED, FAILED)
    priority: ResourcePriority
    backend: ExecutionBackend
    estimated_duration: float
    actual_duration: Optional[float] = None
    error_message: Optional[str] = None
    result_location: Optional[str] = None

# Clase principal para la representación de momentum cuántico
class QuantumMomentumRepresentation:
    """
    Representa y manipula un estado cuántico, enfocándose en las representaciones
    de posición y momentum (base computacional y base de Fourier).
    Permite aplicar compuertas, realizar mediciones y persistir el estado.
    """
    METADATA_FILENAME = "metadata.json"
    STATE_VECTOR_FILENAME = "state_vector.npy"
    CHECKPOINT_PREFIX = "checkpoint_"

    def __init__(self,
                 num_qubits: int = DEFAULT_QUBIT_COUNT,
                 id: Optional[str] = None,
                 backend: ExecutionBackend = ExecutionBackend.SIMULATION,
                 noise_model: Optional[Any] = None): # 'Any' es placeholder para NoiseModel de Qiskit
        if not isinstance(num_qubits, int) or not (0 < num_qubits <= MAX_QUBIT_COUNT):
            raise ValueError(f"El número de qubits ({num_qubits}) debe ser un entero entre 1 y {MAX_QUBIT_COUNT}")

        self.num_qubits = num_qubits
        self.dimension = 2**num_qubits
        self._position_space: Optional[np.ndarray] = None # Representación en base computacional |x>
        self._momentum_space: Optional[np.ndarray] = None # Representación en base de Fourier |p>
        self.id = id if id else str(uuid.uuid4())
        self.backend = backend # Actualmente solo afecta la simulación (ruido)
        self.noise_model = noise_model # Para usar con simuladores Qiskit Aer

        self.metadata = QuantumStateMetadata(
            id=self.id,
            creation_timestamp=time.time(),
            last_modified=time.time(),
            num_qubits=num_qubits,
            state_type=StateType.PURE, # Estado inicial (antes de cargar)
            is_persistent=False
        )
        self._lock = threading.RLock() # Reentrant lock para operaciones seguras en hilos
        self._dirty = False # Flag para indicar si el estado ha cambiado desde el último guardado/checkpoint
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._checkpoint_stop_event = threading.Event()

        # Inicializar estado a |0...0> por defecto
        self.reset_state()

        logger.info(f"Instancia QuantumMomentumRepresentation creada [id={self.id}, qubits={num_qubits}, backend={backend.name}]")

    @property
    def position_space(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._position_space

    @property
    def momentum_space(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._momentum_space is None and self._position_space is not None:
                self._calculate_momentum_space_internal()
            return self._momentum_space

    def reset_state(self):
        """Inicializa el estado a |0...0>."""
        with self._lock:
            self._position_space = np.zeros(self.dimension, dtype=complex)
            self._position_space[0] = 1.0
            self._momentum_space = None # Borrar caché de momentum
            self._update_metadata_on_change()
            self._dirty = True
            logger.info(f"Estado reseteado a |0...0> [id={self.id}]")

    def load_superposition_state(self, state_vector: Union[np.ndarray, Dict[str, complex], List[complex]]) -> None:
        """
        Carga un estado cuántico desde un vector numpy, diccionario o lista.

        Args:
            state_vector: El vector de estado a cargar.
                          - np.ndarray: Vector complejo de tamaño 2^N.
                          - Dict[str, complex]: Mapa de estados base ("010") a amplitudes.
                          - List[complex]: Lista de amplitudes complejas.

        Raises:
            InvalidStateError: Si el estado proporcionado no es válido o no se puede normalizar.
            ValueError: Si el tipo de entrada es incorrecto.
        """
        with self._lock:
            loaded_vector = None
            if isinstance(state_vector, np.ndarray):
                if state_vector.shape != (self.dimension,):
                    raise InvalidStateError(f"El vector Numpy tiene dimensiones incorrectas ({state_vector.shape}). Se esperaba ({self.dimension},)")
                if not np.issubdtype(state_vector.dtype, np.complexfloating):
                     logger.warning(f"El vector Numpy no es complejo. Convirtiendo a complejo.")
                     state_vector = state_vector.astype(complex)
                loaded_vector = state_vector

            elif isinstance(state_vector, dict):
                loaded_vector = np.zeros(self.dimension, dtype=complex)
                if not state_vector:
                    raise InvalidStateError("El diccionario de estado está vacío")
                key_len = len(next(iter(state_vector.keys())))
                if key_len != self.num_qubits:
                     raise InvalidStateError(f"Las claves deben tener {self.num_qubits} bits (encontrado: {key_len})")
                for basis_state, amplitude in state_vector.items():
                    if len(basis_state) != self.num_qubits or not all(c in '01' for c in basis_state):
                        raise InvalidStateError(f"Clave inválida: '{basis_state}'")
                    index = int(basis_state, 2)
                    loaded_vector[index] = complex(amplitude)

            elif isinstance(state_vector, list):
                if len(state_vector) != self.dimension:
                     raise InvalidStateError(f"La lista tiene longitud incorrecta ({len(state_vector)}). Se esperaba {self.dimension}")
                loaded_vector = np.array(state_vector, dtype=complex)

            else:
                raise ValueError("Tipo de state_vector no soportado. Usar np.ndarray, dict o list.")

            # Verificar y normalizar
            norm = np.linalg.norm(loaded_vector)
            if np.isclose(norm, 0.0):
                raise InvalidStateError("El estado proporcionado tiene norma cero.")
            if not np.isclose(norm, 1.0, atol=1e-7):
                logger.warning(f"Estado no normalizado (norma={norm:.4f}). Normalizando automáticamente.")
                loaded_vector = loaded_vector / norm
                # Verificar de nuevo después de normalizar
                if not np.isclose(np.linalg.norm(loaded_vector), 1.0, atol=1e-7):
                     raise InvalidStateError("Falló la normalización del estado.")

            self._position_space = loaded_vector
            self._momentum_space = None # Forzar recálculo si se accede
            self._update_metadata_on_change()
            self._dirty = True
            logger.info(f"Estado cargado y normalizado exitosamente [id={self.id}]")

    def _update_metadata_on_change(self):
        """Actualiza metadatos relevantes cuando el estado cambia."""
        self.metadata.last_modified = time.time()
        if self._position_space is not None:
            self._update_state_type()
            # Podríamos añadir cálculo de fidelidad si hubiera un estado de referencia
            # self.metadata.fidelity_history.append((time.time(), current_fidelity))
        else:
            self.metadata.state_type = StateType.PURE # O algún estado inválido/vacío?

    def _update_state_type(self):
        """Determina el tipo de estado (Simplificado)."""
        if self._position_space is None: return

        is_superposition = self._check_superposition()
        is_entangled = self._check_entanglement() # Costoso, hacer solo si es necesario

        if is_entangled:
            self.metadata.state_type = StateType.ENTANGLED
        elif is_superposition:
            self.metadata.state_type = StateType.SUPERPOSITION
        else:
            # Es un estado base único (no en superposición ni entrelazado)
            self.metadata.state_type = StateType.PURE

        # Calcular/actualizar métricas de entrelazamiento
        if is_entangled or self.num_qubits >= 2: # Calcular siempre para >= 2 qubits
             self.metadata.entanglement_metrics = self._calculate_entanglement_metrics()
        else:
             self.metadata.entanglement_metrics = {} # Limpiar si es 1 qubit

    def _check_superposition(self) -> bool:
        """Verifica si el estado es una superposición de más de un estado base."""
        if self._position_space is None: return False
        significant_amplitudes = np.sum(np.abs(self._position_space)**2 > 1e-9) # Usar probabilidad
        return significant_amplitudes > 1

    def _check_entanglement(self) -> bool:
        """Verifica si el estado está entrelazado usando la pureza de subsistemas (Simplificado)."""
        if self._position_space is None or self.num_qubits < 2:
            return False
        if not HAS_QISKIT:
            logger.warning("Qiskit no disponible, no se puede calcular entrelazamiento robustamente.")
            # Heurística muy simple: si hay más de 2^k + 2^(n-k) - 1 términos no cero, es probable que esté entrelazado? No fiable.
            return False # No podemos verificar sin Qiskit para partial_trace

        try:
            # Verificar pureza de la traza parcial sobre el primer qubit
            # rho = np.outer(self.position_space, np.conjugate(self.position_space)) # Matriz densidad completa
            qargs_keep = list(range(1, self.num_qubits)) # Qubits a mantener (todos menos el 0)
            # ¡Cuidado! partial_trace espera dims (2, 2, ..., 2)
            dims = [2] * self.num_qubits
            rho_reduced = partial_trace(self._position_space, qargs=[0], dims=dims).data # Trazar el qubit 0
            
            purity = np.real(np.trace(np.dot(rho_reduced, rho_reduced)))

            # Si la pureza es significativamente menor que 1, está entrelazado
            return not np.isclose(purity, 1.0, atol=1e-7)
        except Exception as e:
            logger.error(f"Error al verificar entrelazamiento: {e}")
            return False # Asumir no entrelazado si hay error

    def _calculate_entanglement_metrics(self) -> Dict[str, float]:
        """Calcula métricas de entrelazamiento como entropía y pureza."""
        if self._position_space is None or self.num_qubits < 1: return {}
        if not HAS_QISKIT: return {} # Necesario para partial_trace y entropy

        metrics = {}
        try:
            # Entropía de Von Neumann (del estado completo, no mide entrelazamiento directamente)
            # Para medir entrelazamiento, necesitamos entropía de un subsistema
            if self.num_qubits >= 2:
                dims = [2] * self.num_qubits
                rho_reduced_0 = partial_trace(self._position_space, qargs=[0], dims=dims).data
                ent_entropy = entropy(rho_reduced_0, base=np.exp(1)) # Usar base e o 2
                metrics['entanglement_entropy'] = ent_entropy # Entropía del primer qubit reducido
                
                # Pureza del subsistema (relacionado con la entropía)
                purity = np.real(np.trace(np.dot(rho_reduced_0, rho_reduced_0)))
                metrics['subsystem_purity'] = purity

            # Concurrencia (solo para 2 qubits)
            if self.num_qubits == 2:
                 # psi = a|00> + b|01> + c|10> + d|11>
                 a, b, c, d = self._position_space[0], self._position_space[1], self._position_space[2], self._position_space[3]
                 concurrence = 2 * np.abs(a * d - b * c)
                 metrics['concurrence'] = concurrence

        except Exception as e:
            logger.warning(f"Error calculando métricas de entrelazamiento: {e}")

        return metrics

    def _calculate_momentum_space_internal(self) -> None:
        """Calcula la representación de momentum (FFT) internamente."""
        if self._position_space is None:
            # No debería ocurrir si se llama desde la propiedad momentum_space
            raise InvalidStateError("Espacio de posición no definido.")
        try:
            # FFT normalizada
            self._momentum_space = fft(self._position_space, norm="ortho")
            logger.debug(f"Espacio de momentum calculado [id={self.id}]")
        except Exception as e:
            logger.error(f"Error en FFT para espacio de momentum: {e}")
            self._momentum_space = None
            raise # Relanzar la excepción

    def _calculate_position_space_internal(self) -> None:
        """Calcula la representación de posición (IFFT) internamente."""
        if self._momentum_space is None:
             raise InvalidStateError("Espacio de momentum no definido.")
        try:
            # IFFT normalizada
            self._position_space = ifft(self._momentum_space, norm="ortho")
            # Verificar norma después de IFFT por errores numéricos
            norm = np.linalg.norm(self._position_space)
            if not np.isclose(norm, 1.0, atol=1e-7):
                logger.warning(f"Re-normalizando después de IFFT (norma={norm:.4f})")
                self._position_space /= norm
            logger.debug(f"Espacio de posición calculado desde momentum [id={self.id}]")
        except Exception as e:
            logger.error(f"Error en IFFT para espacio de posición: {e}")
            self._position_space = None
            raise

    def get_momentum_probabilities(self) -> np.ndarray:
        """Obtiene las probabilidades de medir cada estado base en el espacio de momentum."""
        with self._lock:
            mom_space = self.momentum_space # Usa la propiedad para asegurar cálculo
            if mom_space is None:
                raise InvalidStateError("No se pudo calcular el espacio de momentum.")
            probs = np.abs(mom_space)**2
            # Asegurar que sumen 1 (debido a errores de punto flotante)
            return probs / np.sum(probs)


    def get_position_probabilities(self) -> np.ndarray:
        """Obtiene las probabilidades de medir cada estado base en el espacio de posición."""
        with self._lock:
            if self._position_space is None:
                raise InvalidStateError("No hay estado cargado en el espacio de posición.")
            probs = np.abs(self._position_space)**2
            # Asegurar que sumen 1
            return probs / np.sum(probs)

    def measure_in_position_basis(self, num_shots: int = 1024) -> Dict[str, int]:
        """Simula mediciones en la base computacional (posición)."""
        with self._lock:
            if self._position_space is None: raise InvalidStateError("Estado no cargado.")
            if num_shots <= 0: raise ValueError("num_shots debe ser positivo.")

            probs = self.get_position_probabilities()
            indices = np.arange(self.dimension)
            try:
                measured_indices = np.random.choice(indices, size=num_shots, p=probs)
                results = {}
                for idx in measured_indices:
                    bit_str = format(idx, f'0{self.num_qubits}b')
                    results[bit_str] = results.get(bit_str, 0) + 1
                logger.info(f"Medición simulada (posición) [id={self.id}, shots={num_shots}]")
                return results
            except ValueError as e: # Puede ocurrir si las probabilidades no suman 1 exacto
                 logger.error(f"Error en np.random.choice (probs sum={np.sum(probs)}): {e}")
                 # Intentar normalizar de nuevo aquí si es necesario
                 raise InvalidStateError(f"Error en la distribución de probabilidad: {e}")
            except Exception as e:
                 logger.error(f"Error durante la medición de posición: {e}")
                 raise

    def measure_in_momentum_basis(self, num_shots: int = 1024) -> Dict[str, int]:
        """Simula mediciones en la base de Fourier (momentum)."""
        with self._lock:
            mom_space = self.momentum_space # Calcula si es necesario
            if mom_space is None: raise InvalidStateError("Estado no cargado o momentum no calculable.")
            if num_shots <= 0: raise ValueError("num_shots debe ser positivo.")

            probs = self.get_momentum_probabilities()
            indices = np.arange(self.dimension)
            try:
                measured_indices = np.random.choice(indices, size=num_shots, p=probs)
                results = {}
                for idx in measured_indices:
                    # La base de momentum también se etiqueta con índices 0..2^N-1
                    bit_str = format(idx, f'0{self.num_qubits}b')
                    results[bit_str] = results.get(bit_str, 0) + 1
                logger.info(f"Medición simulada (momentum) [id={self.id}, shots={num_shots}]")
                return results
            except ValueError as e:
                 logger.error(f"Error en np.random.choice (probs sum={np.sum(probs)}): {e}")
                 raise InvalidStateError(f"Error en la distribución de probabilidad de momentum: {e}")
            except Exception as e:
                 logger.error(f"Error durante la medición de momentum: {e}")
                 raise

    def apply_gate(self, gate_name: str, target_qubit: Union[int, List[int]],
                   control_qubit: Optional[Union[int, List[int]]] = None,
                   params: Optional[List[float]] = None) -> None:
        """
        Aplica una compuerta cuántica al estado actual usando simulación Qiskit.

        Args:
            gate_name (str): Nombre de la compuerta (e.g., "h", "cx", "rz"). Case-insensitive.
            target_qubit (int o List[int]): Índice(s) del qubit(s) objetivo.
            control_qubit (int o List[int], optional): Índice(s) del qubit(s) de control.
            params (list[float], optional): Parámetros para compuertas parametrizadas (e.g., ángulo para Rz).

        Raises:
            InvalidStateError: Si no hay estado cargado o Qiskit no está disponible.
            ValueError: Si los parámetros de la compuerta son inválidos.
            NotImplementedError: Si la compuerta no está soportada.
        """
        if not HAS_QISKIT:
            raise InvalidStateError("Qiskit es necesario para aplicar compuertas.")

        with self._lock:
            if self._position_space is None:
                raise InvalidStateError("No hay estado cargado para aplicar compuerta.")

            # Validar qubits
            targets = [target_qubit] if isinstance(target_qubit, int) else target_qubit
            controls = []
            if control_qubit is not None:
                controls = [control_qubit] if isinstance(control_qubit, int) else control_qubit

            all_qubits = targets + controls
            if not all(0 <= q < self.num_qubits for q in all_qubits):
                 raise ValueError(f"Índice de qubit fuera de rango (0 a {self.num_qubits - 1}). Qubits: {all_qubits}")
            if len(set(all_qubits)) != len(all_qubits):
                 raise ValueError(f"Qubits objetivo y de control deben ser únicos. Qubits: {all_qubits}")

            try:
                # Crear circuito y inicializar con el estado actual
                qc = QuantumCircuit(self.num_qubits, name=f"apply_{gate_name}")
                # Usar Statevector para inicialización más robusta
                initial_state = Statevector(self._position_space)
                qc.initialize(initial_state, range(self.num_qubits))

                # Aplicar la compuerta
                gate_func_name = gate_name.lower()
                
                # Intentar obtener la función de compuerta directamente de qc
                if hasattr(qc, gate_func_name):
                    gate_method = getattr(qc, gate_func_name)
                    
                    # Construir argumentos basados en la signatura esperada (simplificado)
                    args = []
                    if params: args.extend(params)
                    if controls: args.extend(controls) # Control qubits suelen ir primero en Qiskit para métodos como mcx
                    args.extend(targets) # Target qubits al final
                    
                    # Ajuste específico para CNOT/CX, CZ
                    if gate_func_name in ["cx", "cnot"]:
                         if len(controls) != 1 or len(targets) != 1: raise ValueError("CX necesita 1 control y 1 target.")
                         gate_method(controls[0], targets[0])
                    elif gate_func_name == "cz":
                         if len(controls) != 1 or len(targets) != 1: raise ValueError("CZ necesita 1 control y 1 target.")
                         gate_method(controls[0], targets[0])
                    # Compuertas generales (H, X, Y, Z, S, T, Rx, Ry, Rz, etc.)
                    else:
                         # Asume que los parámetros (si existen) van primero, luego los qubits
                         q_args = controls + targets # Orden podría variar para compuertas multicontrol
                         if params:
                            gate_method(*params, *q_args) # Ej: rz(angle, qubit)
                         else:
                            gate_method(*q_args) # Ej: h(qubit), x(qubit)

                else:
                     raise NotImplementedError(f"Compuerta '{gate_name}' no reconocida o soportada directamente.")

                # Simular para obtener el nuevo estado (sin ruido por ahora)
                # TODO: Integrar self.noise_model si está presente
                backend = Aer.get_backend('statevector_simulator')
                # Transpilar podría ser necesario para backends reales, no tanto para statevector
                # transpiled_qc = transpile(qc, backend)
                job = execute(qc, backend, shots=1) # Shots irrelevante para statevector
                result = job.result()
                final_statevector = result.get_statevector(qc)

                # Actualizar estado y metadatos
                self._position_space = final_statevector.data # .data da el numpy array
                self._momentum_space = None # Invalidar caché de momentum
                self._update_metadata_on_change()
                self._dirty = True
                
                logger.info(f"Compuerta {gate_name.upper()} aplicada [id={self.id}, target={targets}, control={controls}, params={params}]")

            except ImportError:
                 raise InvalidStateError("Qiskit Aer backend no disponible.")
            except Exception as e:
                 logger.error(f"Error al aplicar compuerta {gate_name}: {e}")
                 # Podríamos intentar restaurar el estado anterior si falló?
                 raise # Relanzar para que el llamador maneje el error

    def apply_qft(self, inverse=False) -> None:
        """
        Aplica la Transformada Cuántica de Fourier (QFT) o su inversa al estado.
        Nota: Esto actualiza directamente position_space y momentum_space.
        """
        with self._lock:
            if self._position_space is None:
                raise InvalidStateError("No hay estado cargado para aplicar QFT.")

            # Usar la relación directa FFT/IFFT que ya implementamos
            # QFT |x> = 1/sqrt(N) * sum_p exp(2*pi*i*x*p/N) |p>  (FFT)
            # IQFT |p> = 1/sqrt(N) * sum_x exp(-2*pi*i*x*p/N) |x> (IFFT)
            # Si self.momentum_space es la FFT de self.position_space
            
            pos_backup = np.copy(self._position_space)
            mom_backup = self.momentum_space # Calcula si es necesario

            if mom_backup is None: # No debería ocurrir si pos_space existe
                 raise InvalidStateError("No se pudo calcular momentum space para QFT.")

            if not inverse:
                # Aplicar QFT: el nuevo estado de posición es el viejo de momentum (normalizado)
                self._position_space = mom_backup
                # El nuevo momentum es la FFT del nuevo estado de posición, que es FFT(FFT(viejo_pos))
                # FFT(FFT(f(x)))[k] = f(-x)[k] (con normalización)
                # Esto significa que el nuevo momentum es el viejo estado de posición invertido en índice (y conjugado?)
                # Usaremos IFFT(viejo_pos) que es más directo
                self._momentum_space = ifft(pos_backup, norm="ortho")
                op_name = "QFT"
            else:
                # Aplicar IQFT: el nuevo estado de posición es la IFFT del viejo pos = viejo mom
                self._position_space = ifft(pos_backup, norm="ortho")
                # El nuevo momentum es la FFT del nuevo estado de posición = FFT(IFFT(viejo_pos)) = viejo_pos
                self._momentum_space = pos_backup
                op_name = "IQFT"

            # Verificar norma por si acaso
            norm = np.linalg.norm(self._position_space)
            if not np.isclose(norm, 1.0, atol=1e-7):
                logger.warning(f"Re-normalizando después de {op_name} (norma={norm:.4f})")
                self._position_space /= norm
                self._momentum_space = None # Recalcular momentum si pos cambió

            self._update_metadata_on_change()
            self._dirty = True
            logger.info(f"{op_name} aplicada [id={self.id}]")

    def plot_position_distribution(self, save_plot=True) -> Figure:
        """Genera y opcionalmente guarda una gráfica de la distribución de posición."""
        with self._lock:
            if self._position_space is None: raise InvalidStateError("Estado no cargado.")
            probs = self.get_position_probabilities()
            
            fig, ax = plt.subplots(figsize=(12, 7))
            indices = np.arange(self.dimension)
            labels = [format(i, f'0{self.num_qubits}b') for i in indices]
            
            ax.bar(indices, probs, tick_label=labels)
            ax.set_xlabel('Estado Base (Posición)')
            ax.set_ylabel('Probabilidad')
            ax.set_title(f'Distribución de Probabilidad de Posición (N={self.num_qubits}, ID: {self.id[:8]})')
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=90, fontsize=8 if self.num_qubits > 6 else 10)
            plt.tight_layout()

            if save_plot:
                filename = f"position_dist_{self.id}_{int(time.time())}.png"
                filepath = os.path.join(VIS_DIR, filename)
                try:
                    fig.savefig(filepath)
                    logger.info(f"Visualización de posición guardada: {filepath}")
                except Exception as e:
                    logger.error(f"Error al guardar gráfica de posición: {e}")
            return fig # Devolver la figura para posible uso interactivo/display

    def plot_momentum_distribution(self, save_plot=True) -> Figure:
        """Genera y opcionalmente guarda una gráfica de la distribución de momentum."""
        with self._lock:
            mom_space = self.momentum_space # Calcula si es necesario
            if mom_space is None: raise InvalidStateError("Estado no cargado o momentum no calculable.")
            probs = self.get_momentum_probabilities()

            fig, ax = plt.subplots(figsize=(12, 7))
            indices = np.arange(self.dimension)
            # La base de momentum también se indexa 0..N-1
            labels = [format(i, f'0{self.num_qubits}b') for i in indices]

            ax.bar(indices, probs, tick_label=labels, color='orange')
            ax.set_xlabel('Estado Base (Momentum)')
            ax.set_ylabel('Probabilidad')
            ax.set_title(f'Distribución de Probabilidad de Momentum (N={self.num_qubits}, ID: {self.id[:8]})')
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=90, fontsize=8 if self.num_qubits > 6 else 10)
            plt.tight_layout()

            if save_plot:
                filename = f"momentum_dist_{self.id}_{int(time.time())}.png"
                filepath = os.path.join(VIS_DIR, filename)
                try:
                    fig.savefig(filepath)
                    logger.info(f"Visualización de momentum guardada: {filepath}")
                except Exception as e:
                    logger.error(f"Error al guardar gráfica de momentum: {e}")
            return fig

    # --- Persistencia ---
    
    def get_state_directory(self) -> str:
        """Devuelve la ruta al directorio específico para este estado."""
        return os.path.join(STATE_DIR, self.id)

    def save_state(self, description: Optional[str] = None) -> None:
        """
        Guarda el estado cuántico actual y sus metadatos en disco.

        Args:
            description (str, optional): Una descripción opcional para añadir a los metadatos.
        
        Raises:
            PersistenceError: Si ocurre un error durante el guardado.
            InvalidStateError: Si el estado no está cargado.
        """
        with self._lock:
            if self._position_space is None:
                raise InvalidStateError("No hay estado para guardar.")

            state_dir = self.get_state_directory()
            os.makedirs(state_dir, exist_ok=True)
            
            metadata_path = os.path.join(state_dir, self.METADATA_FILENAME)
            state_vector_path = os.path.join(state_dir, self.STATE_VECTOR_FILENAME)
            
            try:
                # Actualizar metadatos antes de guardar
                self.metadata.is_persistent = True
                if description:
                    self.metadata.description = description
                self._update_metadata_on_change() # Asegura que last_modified y tipo estén actualizados

                # Guardar metadatos como JSON
                with open(metadata_path, 'w') as f:
                    json.dump(self.metadata.to_dict(), f, indent=4)
                    
                # Guardar el vector de estado (posición) usando numpy
                np.save(state_vector_path, self._position_space)
                
                self._dirty = False # Marcar como no sucio después de guardar
                logger.info(f"Estado guardado exitosamente [id={self.id}, path={state_dir}]")

            except IOError as e:
                logger.error(f"Error de I/O al guardar estado: {e}")
                raise PersistenceError(f"No se pudo guardar el estado en {state_dir}: {e}") from e
            except Exception as e:
                logger.error(f"Error inesperado al guardar estado: {e}")
                raise PersistenceError(f"Error al guardar estado {self.id}: {e}") from e

    @classmethod
    def load_state(cls, state_id: str) -> 'QuantumMomentumRepresentation':
        """
        Carga un estado cuántico previamente guardado desde el disco.

        Args:
            state_id (str): El ID único del estado a cargar.

        Returns:
            QuantumMomentumRepresentation: La instancia del estado cargado.

        Raises:
            PersistenceError: Si el estado no se encuentra o hay un error al cargar.
            InvalidStateError: Si los datos cargados son inválidos.
        """
        state_dir = os.path.join(STATE_DIR, state_id)
        metadata_path = os.path.join(state_dir, cls.METADATA_FILENAME)
        state_vector_path = os.path.join(state_dir, cls.STATE_VECTOR_FILENAME)

        if not os.path.isdir(state_dir) or not os.path.exists(metadata_path) or not os.path.exists(state_vector_path):
            raise PersistenceError(f"No se encontró el estado guardado con ID {state_id} en {state_dir}")

        try:
            # Cargar metadatos
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            metadata = QuantumStateMetadata.from_dict(metadata_dict)

            # Cargar vector de estado
            position_space = np.load(state_vector_path)

            # Crear nueva instancia y poblarla
            instance = cls(num_qubits=metadata.num_qubits, id=metadata.id) # Usar datos de metadata
            instance.metadata = metadata # Sobrescribir metadata por defecto
            
            # Validar dimensiones cargadas
            expected_dim = 2**instance.num_qubits
            if position_space.shape != (expected_dim,):
                raise InvalidStateError(f"Dimensiones del vector de estado cargado ({position_space.shape}) no coinciden con num_qubits={instance.num_qubits} ({expected_dim},)")

            instance._position_space = position_space
            instance._momentum_space = None # Forzar recálculo al acceder
            instance.metadata.is_persistent = True # Asegurar que esté marcado como persistente
            instance._dirty = False # Estado recién cargado no está sucio

            logger.info(f"Estado cargado exitosamente [id={instance.id}, path={state_dir}]")
            return instance

        except (IOError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Error al cargar estado desde {state_dir}: {e}")
            raise PersistenceError(f"No se pudo cargar el estado {state_id}: {e}") from e
        except Exception as e:
            logger.error(f"Error inesperado al cargar estado {state_id}: {e}")
            raise PersistenceError(f"Error inesperado al cargar estado {state_id}: {e}") from e

    # --- Checkpointing ---

    def _get_checkpoint_path(self) -> str:
         """Obtiene la ruta para el archivo de checkpoint."""
         return os.path.join(CHECKPOINT_DIR, f"{self.CHECKPOINT_PREFIX}{self.id}.npz")

    def _save_checkpoint(self) -> bool:
        """Guarda un checkpoint del estado si está marcado como 'dirty'."""
        with self._lock:
            if not self._dirty or self._position_space is None or not self.metadata.is_persistent:
                # No guardar si no hay cambios, no hay estado, o no es persistente
                return False
                
            checkpoint_path = self._get_checkpoint_path()
            temp_checkpoint_path = checkpoint_path + ".tmp"

            try:
                # Guardar estado y metadatos esenciales en un archivo .npz (eficiente)
                metadata_dict = self.metadata.to_dict()
                np.savez_compressed(temp_checkpoint_path,
                                    position_space=self._position_space,
                                    metadata_json=json.dumps(metadata_dict))

                # Renombrar atómicamente (si es posible en el OS)
                os.replace(temp_checkpoint_path, checkpoint_path)

                self.metadata.last_checkpoint_time = time.time()
                self._dirty = False # Marcar como limpio después del checkpoint
                logger.info(f"Checkpoint guardado para el estado [id={self.id}] en {checkpoint_path}")
                return True
            except Exception as e:
                logger.error(f"Error al guardar checkpoint para {self.id}: {e}")
                # Limpiar archivo temporal si existe
                if os.path.exists(temp_checkpoint_path):
                    try: os.remove(temp_checkpoint_path)
                    except OSError: pass
                return False # Indicar fallo

    def _load_from_checkpoint(self) -> bool:
        """Intenta cargar el estado desde el último checkpoint si existe."""
        checkpoint_path = self._get_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return False # No hay checkpoint

        with self._lock:
            try:
                data = np.load(checkpoint_path, allow_pickle=False) # Evitar pickle por seguridad
                
                position_space = data['position_space']
                metadata_json = data['metadata_json'].item() # .item() para obtener el string del array 0-d
                metadata_dict = json.loads(metadata_json)
                metadata = QuantumStateMetadata.from_dict(metadata_dict)

                # Validar consistencia básica
                if metadata.id != self.id or metadata.num_qubits != self.num_qubits:
                    logger.error(f"Inconsistencia en checkpoint {self.id}. Metadata no coincide. No se cargará.")
                    return False
                
                expected_dim = 2**self.num_qubits
                if position_space.shape != (expected_dim,):
                     logger.error(f"Dimensiones inválidas en checkpoint {self.id}. No se cargará.")
                     return False

                # Cargar estado y metadata
                self._position_space = position_space
                self.metadata = metadata
                self._momentum_space = None # Recalcular
                self._dirty = False # Recién cargado
                
                # Comparar tiempo de modificación con el checkpoint?
                # Si el archivo persistente es más nuevo que el checkpoint, ¿qué hacer?
                # Por ahora, el checkpoint tiene prioridad si se llama a esta función.

                logger.info(f"Estado restaurado desde checkpoint [id={self.id}]")
                return True

            except Exception as e:
                logger.error(f"Error al cargar desde checkpoint {self.id}: {e}")
                return False

    def _checkpoint_worker(self):
        """Función ejecutada por el hilo de checkpointing."""
        logger.info(f"Hilo de checkpoint iniciado para estado [id={self.id}] con intervalo {CHECKPOINT_INTERVAL}s")
        while not self._checkpoint_stop_event.wait(CHECKPOINT_INTERVAL):
            try:
                if self.metadata.is_persistent: # Solo hacer checkpoint de estados persistentes
                    saved = self._save_checkpoint()
                    # if saved: logger.debug(f"Checkpoint realizado para {self.id}")
            except Exception as e:
                # Capturar cualquier excepción inesperada en el worker
                logger.error(f"Error fatal en hilo de checkpoint para {self.id}: {e}")
                # ¿Debería detenerse el hilo aquí? Podría ser un error transitorio.
                # Por ahora, continuamos.
        logger.info(f"Hilo de checkpoint detenido para estado [id={self.id}]")

    def start_checkpointing(self):
        """Inicia el proceso de checkpointing en segundo plano si no está activo."""
        if CHECKPOINT_INTERVAL <= 0:
             logger.warning("Checkpointing deshabilitado (intervalo <= 0).")
             return

        with self._lock: # Proteger acceso a self._checkpoint_thread
            if self._checkpoint_thread is None or not self._checkpoint_thread.is_alive():
                self._checkpoint_stop_event.clear() # Asegurar que no esté puesto
                self._checkpoint_thread = threading.Thread(
                    target=self._checkpoint_worker,
                    name=f"CheckpointWorker-{self.id[:8]}",
                    daemon=True # Permitir que el programa salga aunque el hilo esté activo
                )
                self._checkpoint_thread.start()
            else:
                logger.warning(f"Intento de iniciar checkpointing cuando ya está activo [id={self.id}]")

    def stop_checkpointing(self, save_final: bool = True):
        """Detiene el hilo de checkpointing y opcionalmente guarda un último checkpoint."""
        logger.info(f"Deteniendo checkpointing para estado [id={self.id}]...")
        if self._checkpoint_thread and self._checkpoint_thread.is_alive():
            self._checkpoint_stop_event.set() # Señalizar al hilo que se detenga
            self._checkpoint_thread.join(timeout=5.0) # Esperar a que termine
            if self._checkpoint_thread.is_alive():
                 logger.warning(f"El hilo de checkpoint para {self.id} no terminó a tiempo.")
            self._checkpoint_thread = None
            logger.info(f"Hilo de checkpoint detenido para [id={self.id}]")
        else:
            logger.info(f"Checkpointing no estaba activo para [id={self.id}]")
            
        if save_final and self.metadata.is_persistent:
             logger.info(f"Guardando checkpoint final para [id={self.id}]...")
             self._save_checkpoint()

    def __del__(self):
        """Asegura que el checkpointing se detenga al destruir el objeto."""
        try:
            self.stop_checkpointing(save_final=False) # No guardar al destruir por defecto
        except Exception as e:
            # Logger podría no estar disponible durante __del__
            print(f"WARN: Error al detener checkpointing durante __del__ para {self.id}: {e}", file=sys.stderr)


# --- Ejemplo de Uso (Fuera de la clase) ---
if __name__ == "__main__":
    logger.info("--- Iniciando Ejemplo QuantumOS Knob ---")

    try:
        # 1. Crear una instancia
        q_state = QuantumMomentumRepresentation(num_qubits=3)
        logger.info(f"Estado inicial (posición): {q_state.position_space}")

        # 2. Cargar un estado de superposición (ej: estado W)
        # |W> = (|100> + |010> + |001>) / sqrt(3)
        w_state_dict = {"100": 1/np.sqrt(3), "010": 1/np.sqrt(3), "001": 1/np.sqrt(3)}
        q_state.load_superposition_state(w_state_dict)
        logger.info(f"Estado W cargado (posición): {np.round(q_state.position_space, 3)}")
        logger.info(f"Tipo de estado: {q_state.metadata.state_type}")
        logger.info(f"Métricas de entrelazamiento: {q_state.metadata.entanglement_metrics}")


        # 3. Calcular y mostrar probabilidades
        pos_probs = q_state.get_position_probabilities()
        mom_probs = q_state.get_momentum_probabilities()
        logger.info(f"Probabilidades de Posición: {np.round(pos_probs, 3)}")
        logger.info(f"Probabilidades de Momentum: {np.round(mom_probs, 3)}")

        # 4. Aplicar una compuerta (si Qiskit está disponible)
        if HAS_QISKIT:
            try:
                logger.info("Aplicando Hadamard al qubit 0...")
                q_state.apply_gate("h", target_qubit=0)
                logger.info(f"Nuevo estado (posición): {np.round(q_state.position_space, 3)}")
                logger.info(f"Nuevo tipo de estado: {q_state.metadata.state_type}")
                logger.info(f"Nuevas métricas: {q_state.metadata.entanglement_metrics}")
                
                logger.info("Aplicando CNOT(0, 1)...")
                q_state.apply_gate("cx", target_qubit=1, control_qubit=0)
                logger.info(f"Estado tras CNOT (posición): {np.round(q_state.position_space, 3)}")
                logger.info(f"Nuevo tipo de estado: {q_state.metadata.state_type}")
                logger.info(f"Nuevas métricas: {q_state.metadata.entanglement_metrics}")

            except (InvalidStateError, ValueError, NotImplementedError) as e:
                logger.error(f"Error al aplicar compuerta: {e}")
        else:
            logger.warning("Qiskit no encontrado, saltando aplicación de compuertas.")

        # 5. Realizar mediciones simuladas
        pos_measurement = q_state.measure_in_position_basis(num_shots=2048)
        mom_measurement = q_state.measure_in_momentum_basis(num_shots=2048)
        logger.info(f"Medición (posición): {pos_measurement}")
        logger.info(f"Medición (momentum): {mom_measurement}")

        # 6. Generar y guardar gráficas
        try:
            fig_pos = q_state.plot_position_distribution(save_plot=True)
            plt.close(fig_pos) # Cerrar figura para liberar memoria
            fig_mom = q_state.plot_momentum_distribution(save_plot=True)
            plt.close(fig_mom)
        except Exception as e:
            logger.error(f"Error al generar/guardar plots: {e}")

        # 7. Persistencia
        state_id = q_state.id
        try:
            logger.info("Guardando estado...")
            q_state.save_state(description="Estado de prueba después de H y CNOT")
            logger.info(f"Estado {state_id} guardado.")

            # Liberar la instancia actual (simulado)
            del q_state
            logger.info("Instancia original eliminada.")
            
            # Cargar el estado guardado
            logger.info(f"Cargando estado {state_id}...")
            loaded_q_state = QuantumMomentumRepresentation.load_state(state_id)
            logger.info(f"Estado cargado. Descripción: {loaded_q_state.metadata.description}")
            logger.info(f"Vector de posición cargado: {np.round(loaded_q_state.position_space, 3)}")
            
            # Verificar que el momentum se puede recalcular
            loaded_mom_probs = loaded_q_state.get_momentum_probabilities()
            logger.info(f"Probabilidades de Momentum (cargado): {np.round(loaded_mom_probs, 3)}")

        except (PersistenceError, InvalidStateError) as e:
            logger.error(f"Error durante la persistencia: {e}")
            
        # 8. Checkpointing (si se cargó el estado)
        if 'loaded_q_state' in locals():
             try:
                 logger.info("Iniciando checkpointing...")
                 loaded_q_state.start_checkpointing()
                 
                 # Simular trabajo y cambios
                 logger.info("Simulando trabajo (espera)...")
                 time.sleep(5)
                 if HAS_QISKIT:
                     loaded_q_state.apply_gate("x", 0)
                     logger.info("Compuerta X aplicada, estado marcado como 'dirty'.")
                 
                 # Esperar para que ocurra un checkpoint (si el intervalo es corto)
                 # O forzar uno manualmente (no implementado, pero _save_checkpoint podría llamarse)
                 logger.info(f"Esperando {CHECKPOINT_INTERVAL + 5}s para posible checkpoint automático...")
                 time.sleep(CHECKPOINT_INTERVAL + 5)

                 logger.info("Deteniendo checkpointing...")
                 loaded_q_state.stop_checkpointing(save_final=True)
                 
                 # Intentar restaurar desde checkpoint (simulación)
                 logger.info("Intentando cargar desde checkpoint...")
                 restored = loaded_q_state._load_from_checkpoint()
                 if restored:
                     logger.info("Estado restaurado desde el último checkpoint.")
                 else:
                      logger.warning("No se pudo restaurar desde checkpoint.")

             except Exception as e:
                 logger.error(f"Error durante el ejemplo de checkpointing: {e}")
             finally:
                # Asegurarse de detener el hilo si aún está vivo
                if 'loaded_q_state' in locals() and loaded_q_state._checkpoint_thread and loaded_q_state._checkpoint_thread.is_alive():
                    loaded_q_state.stop_checkpointing(save_final=False)

    except Exception as e:
        logger.exception("Error fatal en el script principal.") # Imprime traceback

    finally:
        # Limpieza si es necesario (cerrar plots abiertos, etc.)
        plt.close('all') 
        logger.info("--- Ejemplo QuantumOS Knob Finalizado ---")

"""
Documentación:

plot_momentum_distribution Completado: Implementado de forma similar a plot_position_distribution, usando get_momentum_probabilities y guardando la figura.

Entrelazamiento y Métricas:

_check_entanglement: Se usa qiskit.quantum_info.partial_trace para calcular la pureza de un subsistema. Si la pureza < 1, está entrelazado. Requiere Qiskit.

_calculate_entanglement_metrics: Calcula la entropía de entrelazamiento (entropía de Von Neumann del estado reducido) y la concurrencia (para 2 qubits).

Se actualizan las métricas y el tipo de estado (StateType) después de operaciones que modifican el estado.

Representación Interna: Se usan _position_space y _momentum_space como variables internas, accedidas a través de properties (@property). La propiedad momentum_space ahora calcula automáticamente el espacio de momentum si no existe y el de posición sí.

Cálculo FFT/IFFT: Se usa fft(..., norm="ortho") y ifft(..., norm="ortho") para que las transformadas sean unitarias y preserven la norma (más cercano a la QFT teórica).

apply_gate:

Se usan métodos directamente del objeto QuantumCircuit de Qiskit (qc.h(), qc.cx(), qc.rz(), etc.), haciéndolo más flexible.

Maneja listas de qubits para target/control (aunque muchas compuertas Qiskit aún esperan índices individuales).

Validación de índices de qubits.

Usa Statevector(vector).data para obtener el array numpy del resultado.

apply_qft Implementado: Usa la relación directa entre FFT/IFFT y las representaciones de posición/momentum. QFT transforma position_space en momentum_space, e IQFT hace lo inverso.

Persistencia (save_state, load_state):

Guarda los metadatos en metadata.json y el vector de estado (position_space) en state_vector.npy dentro de un directorio específico para el ID del estado (~/.quantumos/states/<state_id>/).

load_state es un @classmethod que reconstruye la instancia desde los archivos guardados.

Se manejan errores de I/O y formato.

Checkpointing:

Se añadió un flag _dirty que se activa cuando el estado cambia.

_save_checkpoint: Guarda el estado y metadatos esenciales en un archivo .npz (comprimido) si _dirty es True y el estado es persistente. Usa renombrado atómico para mayor seguridad.

_load_from_checkpoint: Restaura el estado desde el archivo de checkpoint.

_checkpoint_worker: Función que corre en un hilo separado y llama periódicamente a _save_checkpoint.

start_checkpointing y stop_checkpointing: Métodos para controlar el hilo de checkpointing.

__del__: Intenta detener el hilo de checkpointing cuando el objeto es destruido.

Robustez:

Se usa threading.RLock para proteger el acceso concurrente al estado y metadatos.

Manejo de errores mejorado con excepciones personalizadas y logging detallado.

Validaciones de entrada (tipos, rangos, dimensiones).

Se asegura la normalización del estado después de cargas y transformaciones.

Ejemplo de Uso (if __name__ == "__main__":): Demuestra cómo crear, cargar, manipular, medir, graficar, guardar, cargar y usar checkpointing con la clase.

Qiskit Opcional: Se verifica HAS_QISKIT antes de usar funciones específicas de Qiskit (apply_gate, cálculos de entrelazamiento). Si no está instalado, esas funciones lanzarán un error o se saltarán (en el ejemplo).

"""
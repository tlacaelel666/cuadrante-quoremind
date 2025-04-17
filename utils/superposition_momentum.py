#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuantumOS Knob: Sistema Híbrido Cuántico-Bayesiano con Gestión de Momentum en Superposición

Este componente proporciona:
1. Interfaz para la gestión de estados cuánticos en un SO simulado.
2. Monitoreo y manipulación del momentum cuántico (representación de Fourier).
3. Aplicación de compuertas cuánticas (vía Qiskit Statevector simulator).
4. Persistencia (guardado/carga) y checkpointing de estados cuánticos.
5. APIs básicas para la integración con otros componentes del SO.

Autor: Jacobo Tlacaelel Mina Rodríguez
Colaboradores: Equipo QuantumOS
Versión: 0.9.2-beta
Fecha: 01/04/2025 (Revisado)
"""

import numpy as np
# Tensorflow importado originalmente, pero no usado en esta clase.
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model, save_model
# from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import scipy.stats as stats # No usado directamente aquí
from scipy.fft import fft, ifft
import pickle # Considerar alternativas más seguras si se usa para datos no propios. No usado actualmente.
import json
import os
import logging
import time
import threading
# import queue # No usado directamente aquí
import uuid
import warnings
# import hashlib # No usado directamente aquí
import sys # Para stderr en __del__
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum, auto

# Configuración de Matplotlib para headless/backend no interactivo
try:
    import matplotlib
    matplotlib.use('Agg')  # Configuración para headless ANTES de importar pyplot
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    warnings.warn("Matplotlib no está instalado. La visualización estará deshabilitada.")
    HAS_MATPLOTLIB = False
    Figure = type(None) # Placeholder type

# Dependencias específicas para la parte cuántica (Qiskit)
try:
    # Utilizar qiskit-aer para simuladores
    from qiskit_aer import AerSimulator
    # Clases principales de Qiskit
    from qiskit import QuantumCircuit, transpile, execute
    # Clases para manejo de estados y métricas
    from qiskit.quantum_info import Statevector, state_fidelity, entropy, partial_trace
    # Clases para visualización (requieren estado o cuentas)
    from qiskit.visualization import plot_bloch_multivector, plot_histogram
    # Modelos de ruido (opcional, requiere manejo específico)
    # from qiskit_aer.noise import NoiseModel # Importar si se va a usar
    # Mitigación de errores (requiere qiskit-experiments o similar)
    # from qiskit.ignis.mitigation.measurement import CompleteMeasFitter # Ignis obsoleto

    # Nota: Qiskit 1.0+ ha movido Aer a qiskit_aer y puede tener otros cambios de API.
    # Este código asume una versión razonablemente moderna pero pre-1.0 podría necesitar ajustes.
    # O si usa Qiskit 1.0+, ajustar imports y llamadas (ej. execute).

    HAS_QISKIT = True
except ImportError:
    warnings.warn("Qiskit (qiskit-terra, qiskit-aer) no está instalado. Utilizando modo de simulación limitado (sin aplicación de compuertas ni métricas cuánticas avanzadas).")
    HAS_QISKIT = False

    # --- Clases Dummy si Qiskit no está presente ---
    class DummyQuantumCircuit:
        """Placeholder for Qiskit QuantumCircuit."""
        def __init__(self, num_qubits: int, name: Optional[str] = None):
            self.num_qubits = num_qubits
            self.name = name
            if num_qubits <= 0: raise ValueError("Number of qubits must be positive")
        def initialize(self, *args, **kwargs): pass
        def x(self, *args, **kwargs): pass
        def y(self, *args, **kwargs): pass
        def z(self, *args, **kwargs): pass
        def h(self, *args, **kwargs): pass
        def s(self, *args, **kwargs): pass
        def t(self, *args, **kwargs): pass
        def cx(self, *args, **kwargs): pass
        def cz(self, *args, **kwargs): pass
        def rz(self, *args, **kwargs): pass
        # Añadir otras compuertas si son referenciadas
        def __getattr__(self, name: str) -> Callable:
            # Método genérico para evitar errores si se llama a cualquier compuerta
            def dummy_gate(*args, **kwargs):
                 warnings.warn(f"Qiskit no disponible, llamada a compuerta '{name}' ignorada.")
                 pass
            return dummy_gate

    class DummyAerSimulator:
        """Placeholder for Qiskit AerSimulator."""
        def run(self, circuit, **options):
            raise NotImplementedError("Qiskit AerSimulator no disponible.")

    class DummyStatevector:
        """Placeholder for Qiskit Statevector."""
        def __init__(self, data: np.ndarray):
            self.data = data
        def evolve(self, qc):
             raise NotImplementedError("Qiskit Statevector.evolve no disponible.")

    # Sobrescribir clases/funciones de Qiskit con Dummies
    QuantumCircuit = DummyQuantumCircuit
    if 'AerSimulator' not in locals(): AerSimulator = DummyAerSimulator # type: ignore
    Statevector = DummyStatevector # type: ignore
    # Funciones que requieren Qiskit también deben manejarse o lanzar error
    def state_fidelity(state1, state2, **kwargs): return 1.0 # Dummy
    def entropy(state, **kwargs): return 0.0 # Dummy
    def partial_trace(state, qargs, dims): raise NotImplementedError("Qiskit partial_trace no disponible.")
    def plot_bloch_multivector(state): raise NotImplementedError("Qiskit plot_bloch_multivector no disponible.")
    def plot_histogram(counts): raise NotImplementedError("Qiskit plot_histogram no disponible.")
    def execute(circuit, backend, **options): raise NotImplementedError("Qiskit execute no disponible.")
    # --- Fin Clases Dummy ---


# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler("quantumos_knob.log", mode='a'), # Append mode
        logging.StreamHandler(sys.stdout) # Usar stdout para mejor compatibilidad con PTYs
    ]
)
# Usar un logger específico para esta clase/módulo
logger = logging.getLogger("QuantumOS_Knob")

# Definición de constantes del sistema
PERSISTENCE_DIR = os.environ.get("QUANTUM_OS_DATA", os.path.join(os.path.expanduser("~"), ".quantumos"))
STATE_DIR = os.path.join(PERSISTENCE_DIR, "states")
VIS_DIR = os.path.join(PERSISTENCE_DIR, "visualizations")
CHECKPOINT_DIR = os.path.join(PERSISTENCE_DIR, "checkpoints")
API_VERSION = "v1"
DEFAULT_QUBIT_COUNT = 5
MAX_QUBIT_COUNT = 18 # Límite para simulación local (2^18 ~ 262k amplitudes, ~4MB memoria)
THREAD_POOL_SIZE = 4 # No usado directamente en esta clase, ¿quizás para un gestor de trabajos externo?
CHECKPOINT_INTERVAL = 300  # segundos (5 minutos)
NUMPY_SAVE_PRECISION = 15 # Precisión para guardar floats numpy

# Creación de directorios necesarios
os.makedirs(PERSISTENCE_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Enumeraciones para el sistema
class ResourcePriority(Enum):
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

class ExecutionBackend(Enum):
    SIMULATION_STATEVECTOR = auto() # Ideal simulation
    SIMULATION_DENSITY_MATRIX = auto() # Noise simulation
    SIMULATION_QASM = auto() # Measurement simulation (counts)
    LOCAL_HARDWARE = auto() # Placeholder
    REMOTE_IBM = auto()     # Placeholder
    REMOTE_AZURE = auto()   # Placeholder
    REMOTE_AMAZON = auto()  # Placeholder
    CUSTOM = auto()

class StateType(Enum):
    PURE = auto()
    MIXED = auto() # Requeriría Density Matrix
    ENTANGLED = auto() # Puede ser PURE o MIXED
    SUPERPOSITION = auto() # Puede ser PURE o MIXED, y ENTANGLED o SEPARABLE
    SEPARABLE = auto() # No entrelazado

# Clases de excepción personalizadas
class QuantumOSError(Exception): pass
class ResourceUnavailableError(QuantumOSError): pass
class StateDecoherenceError(QuantumOSError): pass # Necesitaría simulación de ruido
class InvalidStateError(QuantumOSError): pass
class BackendCommunicationError(QuantumOSError): pass
class PersistenceError(QuantumOSError): pass
class QiskitNotAvailableError(QuantumOSError): pass

# Dataclasses para estructuras de datos
@dataclass
class QuantumStateMetadata:
    id: str
    creation_timestamp: float
    last_modified: float
    num_qubits: int
    state_type: StateType # Caracterización del estado actual
    is_persistent: bool = False
    owner_process: Optional[int] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    fidelity_history: List[Tuple[float, float]] = field(default_factory=list) # (timestamp, fidelity vs reference?)
    entanglement_metrics: Dict[str, float] = field(default_factory=dict)
    last_checkpoint_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convertir Enum a string para JSON
        data['state_type'] = self.state_type.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumStateMetadata':
        # Convertir string de Enum de nuevo a Enum
        try:
            state_type_enum = StateType[data['state_type']]
        except KeyError:
            logger.warning(f"Tipo de estado desconocido '{data['state_type']}' en metadata cargada. Usando PURE por defecto.")
            state_type_enum = StateType.PURE
        data['state_type'] = state_type_enum

        # Asegurar compatibilidad si faltan campos en archivos antiguos
        # Crear instancia solo con campos obligatorios primero
        obj = cls(id=data['id'],
                  num_qubits=data['num_qubits'],
                  # Usar valores por defecto si faltan campos básicos
                  creation_timestamp=data.get('creation_timestamp', time.time()),
                  last_modified=data.get('last_modified', time.time()),
                  state_type=state_type_enum)

        # Poblar campos opcionales/nuevos del diccionario
        for key, value in data.items():
            # Solo setear atributos que existen en la clase y no son los obligatorios ya seteados
            if hasattr(obj, key) and key not in ['id', 'num_qubits', 'creation_timestamp', 'last_modified', 'state_type']:
                 # Manejar casos especiales si es necesario (e.g., convertir listas de listas a tuplas)
                if key == 'fidelity_history' and value is not None:
                    # Asegurar que sean tuplas (timestamp, fidelity)
                    try:
                        obj.fidelity_history = [(float(ts), float(fid)) for ts, fid in value]
                    except (TypeError, ValueError):
                        logger.warning(f"Error al convertir fidelity_history para {obj.id}. Reiniciando.")
                        obj.fidelity_history = []

                elif hasattr(obj, key):
                    try:
                        setattr(obj, key, value)
                    except TypeError:
                         logger.warning(f"Error de tipo al setear {key}={value} para {obj.id}. Usando valor por defecto.")
                         # No hacer nada, mantendrá el valor por defecto del dataclass
        return obj

@dataclass
class SystemResources: # Placeholder
    available_qubits: int = MAX_QUBIT_COUNT
    quantum_memory_usage_mb: float = 0.0
    classical_memory_usage_mb: float = 0.0
    active_quantum_processes: int = 0
    backend_status: Dict[str, str] = field(default_factory=dict) # e.g., {"SIMULATION_STATEVECTOR": "AVAILABLE"}
    error_rates: Dict[str, float] = field(default_factory=dict) # Placeholder
    decoherence_times: Dict[str, float] = field(default_factory=dict) # Placeholder

@dataclass
class JobMetadata: # Placeholder
    job_id: str
    state_id: str
    submission_time: float
    status: str # (e.g., PENDING, RUNNING, COMPLETED, FAILED)
    priority: ResourcePriority = ResourcePriority.NORMAL
    backend: ExecutionBackend = ExecutionBackend.SIMULATION_STATEVECTOR
    estimated_duration_sec: Optional[float] = None
    actual_duration_sec: Optional[float] = None
    error_message: Optional[str] = None
    result_location: Optional[str] = None


class QuantumMomentumRepresentation:
    """
    Representa y manipula un estado cuántico puro simulado, enfocándose en las
    representaciones de posición (base computacional) y momentum (base de Fourier).
    Permite aplicar compuertas (usando Qiskit Aer), realizar mediciones simuladas,
    calcular métricas básicas de entrelazamiento y persistir/checkpoint el estado.

    Nota: Actualmente solo maneja estados puros y usa el simulador de vector de
    estados de Qiskit Aer. La simulación de ruido o ejecución en hardware real
    requeriría extensiones significativas.
    """
    METADATA_FILENAME = "metadata.json"
    STATE_VECTOR_FILENAME = "state_vector.npy"
    CHECKPOINT_PREFIX = "checkpoint_"

    def __init__(self,
                 num_qubits: int = DEFAULT_QUBIT_COUNT,
                 id: Optional[str] = None,
                 initial_state: Union[str, np.ndarray, Dict[str, complex], List[complex], None] = None,
                 backend_type: ExecutionBackend = ExecutionBackend.SIMULATION_STATEVECTOR,
                 noise_model: Optional[Any] = None): # 'Any' es placeholder para qiskit_aer.noise.NoiseModel
        """
        Inicializa la representación del estado cuántico.

        Args:
            num_qubits: Número de qubits en el registro.
            id: ID único para el estado. Si es None, se genera uno nuevo.
            initial_state: El estado inicial. Puede ser:
                           - None o "zero": Estado |0...0>. (Default)
                           - "plus": Estado |+...+> (superposición equitativa).
                           - np.ndarray: Vector de estado (complejo, normalizado, tamaño 2^N).
                           - Dict[str, complex]: Amplitudes { "010": amp, ... }.
                           - List[complex]: Lista de amplitudes.
            backend_type: Backend de ejecución preferido (actualmente solo afecta simulación).
            noise_model: Modelo de ruido de Qiskit Aer (no soportado activamente en apply_gate con Statevector).
        """
        if not isinstance(num_qubits, int) or not (0 < num_qubits <= MAX_QUBIT_COUNT):
            raise ValueError(f"El número de qubits ({num_qubits}) debe ser un entero entre 1 y {MAX_QUBIT_COUNT}")

        self.num_qubits = num_qubits
        self.dimension = 2**num_qubits
        self._position_space: Optional[np.ndarray] = None # Representación |x> (vector de estado)
        self._momentum_space: Optional[np.ndarray] = None # Representación |p> (FFT de |x>)
        self.id = id if id else str(uuid.uuid4())
        self.backend_type = backend_type # Guardar para referencia futura
        self.noise_model = noise_model # Guardar, pero advertir si se intenta usar donde no aplica

        self.metadata = QuantumStateMetadata(
            id=self.id,
            creation_timestamp=time.time(),
            last_modified=time.time(),
            num_qubits=num_qubits,
            state_type=StateType.PURE, # Estado inicial antes de cargar/calcular
            is_persistent=False
        )
        # Reentrant lock para operaciones seguras en hilos sobre el estado
        self._lock = threading.RLock()
        # Flag para indicar si el estado ha cambiado desde el último guardado/checkpoint
        self._dirty = False
        # Atributos para el hilo de checkpointing
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._checkpoint_stop_event = threading.Event()

        # Inicializar el estado según 'initial_state'
        if initial_state is None or initial_state == "zero":
            self.reset_state()
        elif initial_state == "plus":
            self._initialize_plus_state()
        else:
            try:
                self.load_superposition_state(initial_state)
            except (InvalidStateError, ValueError) as e:
                logger.error(f"Error al cargar estado inicial proporcionado: {e}. Reiniciando a |0...0>.")
                self.reset_state()

        logger.info(f"Instancia QuantumMomentumRepresentation creada [id={self.id}, qubits={num_qubits}, backend={backend_type.name}]")

    def __repr__(self) -> str:
        """Representación concisa del objeto."""
        with self._lock:
            state_desc = "No cargado"
            if self._position_space is not None:
                 # Mostrar las primeras/últimas amplitudes si es grande
                 if self.dimension > 8:
                     amps = np.round(self._position_space[:4], 3).tolist() + ["..."] + np.round(self._position_space[-4:], 3).tolist()
                 else:
                     amps = np.round(self._position_space, 3).tolist()
                 state_desc = f"Amps: {amps}"

            return (f"<QuantumMomentumRepresentation(id='{self.id[:8]}...', qubits={self.num_qubits}, "
                    f"state_type='{self.metadata.state_type.name}', dirty={self._dirty}, "
                    f"state={state_desc})>")

    # --- Propiedades para acceder al estado ---

    @property
    def position_space(self) -> Optional[np.ndarray]:
        """Devuelve el vector de estado en la base computacional (|x>)."""
        with self._lock:
            # Devolver una copia para evitar modificaciones externas no deseadas?
            # return self._position_space.copy() if self._position_space is not None else None
            # Por eficiencia, devolvemos la referencia interna. El RLock previene race conditions.
            return self._position_space

    @property
    def momentum_space(self) -> Optional[np.ndarray]:
        """
        Devuelve el vector de estado en la base de momentum (|p>, via FFT).
        Calcula la FFT si no está cacheada.
        """
        with self._lock:
            if self._momentum_space is None and self._position_space is not None:
                # Calcula y cachea la representación de momentum si no existe
                self._calculate_momentum_space_internal()
            # return self._momentum_space.copy() if self._momentum_space is not None else None
            return self._momentum_space

    # --- Métodos de inicialización y carga ---

    def reset_state(self):
        """Inicializa (o resetea) el estado a |0...0>."""
        with self._lock:
            self._position_space = np.zeros(self.dimension, dtype=complex)
            self._position_space[0] = 1.0
            self._momentum_space = None # Borrar caché de momentum
            self._update_metadata_on_change() # Actualiza tipo, timestamp
            self._dirty = True # Estado ha cambiado
            logger.info(f"Estado reseteado a |0...0> [id={self.id}]")

    def _initialize_plus_state(self):
        """Inicializa el estado a |+...+>."""
        with self._lock:
            amplitude = 1.0 / np.sqrt(self.dimension)
            self._position_space = np.full(self.dimension, amplitude, dtype=complex)
            self._momentum_space = None # Borrar caché
            self._update_metadata_on_change()
            self._dirty = True
            logger.info(f"Estado inicializado a |+...+> [id={self.id}]")

    def load_superposition_state(self, state_vector: Union[np.ndarray, Dict[str, complex], List[complex]]) -> None:
        """
        Carga un estado cuántico desde un vector numpy, diccionario o lista.
        El estado se valida y normaliza si es necesario.

        Args:
            state_vector: El vector de estado a cargar.
                          - np.ndarray: Vector complejo de tamaño 2^N.
                          - Dict[str, complex]: Mapa de estados base ("010") a amplitudes.
                          - List[complex]: Lista de amplitudes complejas.

        Raises:
            InvalidStateError: Si el estado proporcionado no es válido (dimensión, formato, norma cero)
                               o no se puede normalizar.
            ValueError: Si el tipo de entrada es incorrecto.
        """
        with self._lock:
            loaded_vector: Optional[np.ndarray] = None
            expected_shape = (self.dimension,)

            if isinstance(state_vector, np.ndarray):
                if state_vector.shape != expected_shape:
                    raise InvalidStateError(f"Vector Numpy tiene dimensiones incorrectas {state_vector.shape}. Se esperaba {expected_shape}")
                if not np.issubdtype(state_vector.dtype, np.complexfloating):
                     logger.warning("Vector Numpy no es complejo. Convirtiendo a complejo.")
                     loaded_vector = state_vector.astype(complex)
                else:
                     loaded_vector = state_vector.copy() # Copiar para evitar efectos secundarios

            elif isinstance(state_vector, dict):
                if not state_vector: raise InvalidStateError("Diccionario de estado está vacío.")
                loaded_vector = np.zeros(self.dimension, dtype=complex)
                # Validar longitud de claves una vez
                first_key = next(iter(state_vector.keys()))
                if not isinstance(first_key, str) or len(first_key) != self.num_qubits:
                    raise InvalidStateError(f"Claves del diccionario deben ser strings binarios de longitud {self.num_qubits}")

                for basis_state, amplitude in state_vector.items():
                    if not isinstance(basis_state, str) or len(basis_state) != self.num_qubits or not all(c in '01' for c in basis_state):
                        raise InvalidStateError(f"Clave de diccionario inválida: '{basis_state}'")
                    try:
                        index = int(basis_state, 2)
                        loaded_vector[index] = complex(amplitude)
                    except ValueError:
                         raise InvalidStateError(f"Amplitud inválida '{amplitude}' para estado '{basis_state}'")
                    except IndexError:
                         # Esto no debería ocurrir si num_qubits y dimension son consistentes
                         raise QuantumOSError(f"Índice {index} fuera de rango para dimensión {self.dimension}. Error interno.")

            elif isinstance(state_vector, list):
                if len(state_vector) != self.dimension:
                     raise InvalidStateError(f"Lista tiene longitud incorrecta {len(state_vector)}. Se esperaba {self.dimension}")
                try:
                    loaded_vector = np.array(state_vector, dtype=complex)
                except ValueError:
                     raise InvalidStateError("Elementos de la lista no se pudieron convertir a complejos.")

            else:
                raise ValueError(f"Tipo de state_vector no soportado: {type(state_vector)}. Usar np.ndarray, dict o list.")

            # --- Validación y Normalización ---
            if loaded_vector is None: # No debería ocurrir si la lógica anterior es correcta
                raise InvalidStateError("Fallo interno al procesar el vector de estado.")

            norm = np.linalg.norm(loaded_vector)
            if np.isclose(norm, 0.0, atol=1e-9):
                raise InvalidStateError("El estado proporcionado tiene norma (casi) cero.")

            if not np.isclose(norm, 1.0, rtol=1e-6, atol=1e-9):
                logger.warning(f"Estado no normalizado (norma={norm:.6f}). Normalizando automáticamente.")
                loaded_vector = loaded_vector / norm
                # Verificar de nuevo después de normalizar
                new_norm = np.linalg.norm(loaded_vector)
                if not np.isclose(new_norm, 1.0, rtol=1e-6, atol=1e-9):
                     # Podría fallar si el vector original era enorme y causó overflow/underflow
                     logger.error(f"Falló la normalización del estado (norma después={new_norm:.6f}).")
                     raise InvalidStateError("Falló la normalización del estado.")

            # --- Carga final ---
            self._position_space = loaded_vector
            self._momentum_space = None # Forzar recálculo si se accede
            self._update_metadata_on_change()
            self._dirty = True
            logger.info(f"Estado cargado y normalizado exitosamente [id={self.id}]")

    # --- Metadatos y Cálculo Interno ---

    def _update_metadata_on_change(self):
        """Actualiza metadatos relevantes (timestamp, tipo, métricas) cuando el estado cambia."""
        self.metadata.last_modified = time.time()
        if self._position_space is not None:
            # Calcular tipo y métricas puede ser costoso, hacerlo aquí
            self._update_state_type_and_metrics()
        else:
            # Si no hay estado, resetear tipo y métricas
            self.metadata.state_type = StateType.PURE # O un tipo "Empty"? PURE es razonable para estado 0
            self.metadata.entanglement_metrics = {}

    def _update_state_type_and_metrics(self):
        """Determina el tipo de estado (Pure, Superposition, Entangled, Separable) y calcula métricas."""
        if self._position_space is None: return # No hacer nada si no hay estado

        # 1. ¿Es Superposición?
        is_superposition = self._check_superposition()

        # 2. ¿Es Entrelazado? (Solo relevante para N >= 2)
        is_entangled = False
        entanglement_metrics = {}
        if self.num_qubits >= 2:
            if HAS_QISKIT:
                try:
                    entanglement_metrics = self._calculate_entanglement_metrics()
                    # Usar la entropía de entrelazamiento como indicador principal
                    # Si la entropía es significativamente > 0, está entrelazado.
                    ent_entropy = entanglement_metrics.get('entanglement_entropy', 0.0)
                    if not np.isclose(ent_entropy, 0.0, atol=1e-7):
                        is_entangled = True
                except QiskitNotAvailableError:
                     logger.warning("Qiskit no disponible, no se pueden calcular métricas de entrelazamiento.")
                except Exception as e:
                     logger.error(f"Error al calcular métricas de entrelazamiento: {e}")
            else:
                 logger.warning("Qiskit no disponible, no se puede determinar entrelazamiento.")

        # 3. Determinar StateType
        if is_entangled:
            self.metadata.state_type = StateType.ENTANGLED
        elif is_superposition:
            # Si es superposición pero no entrelazado (o no se pudo verificar), es separable en superposición
            self.metadata.state_type = StateType.SUPERPOSITION # Podríamos tener SUPERPOSITION_SEPARABLE?
        else:
            # Si no es superposición, es un estado base computacional único (separable)
            self.metadata.state_type = StateType.PURE # Estado base puro |psi> = |comp_basis>

        # Nota: Faltaría StateType.MIXED, que requiere Density Matrix.
        # Nota: Podríamos añadir StateType.SEPARABLE si is_entangled es False y num_qubits >= 2?

        # Actualizar métricas en metadata
        self.metadata.entanglement_metrics = entanglement_metrics


    def _check_superposition(self) -> bool:
        """Verifica si el estado es una superposición de más de un estado base computacional."""
        if self._position_space is None: return False
        # Contar amplitudes cuya probabilidad asociada (|amp|^2) sea mayor que una pequeña tolerancia
        probabilities = np.abs(self._position_space)**2
        significant_amplitudes = np.sum(probabilities > 1e-9)
        return significant_amplitudes > 1

    def _calculate_entanglement_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de entrelazamiento usando Qiskit (entropía, pureza, concurrencia).
        Requiere Qiskit y asume un estado puro.
        """
        if not HAS_QISKIT:
             raise QiskitNotAvailableError("Qiskit necesario para calcular métricas de entrelazamiento.")
        if self._position_space is None or self.num_qubits < 2:
            return {} # No hay entrelazamiento para < 2 qubits

        metrics = {}
        try:
            # Usar Statevector de Qiskit para facilitar cálculos
            qiskit_sv = Statevector(self._position_space)
            dims = tuple([2] * self.num_qubits) # Dimensiones del sistema (2, 2, ..., 2)

            # Calcular entropía de entrelazamiento (Von Neumann entropy of reduced density matrix)
            # Trazamos todos los qubits excepto el primero (qubit 0)
            # Qargs son los qubits a TRAZAR
            try:
                rho_reduced_0 = partial_trace(qiskit_sv, qargs=list(range(1, self.num_qubits))).data
                # Usar base=2 para entropía en bits
                ent_entropy = entropy(rho_reduced_0, base=2)
                # La entropía puede ser ligeramente negativa por errores numéricos, forzar a >= 0
                metrics['entanglement_entropy'] = max(0.0, ent_entropy)
            except Exception as e:
                 logger.warning(f"Error calculando entropía de subsistema 0: {e}")


            # Pureza del subsistema (Tr(rho_reduced^2))
            try:
                # Re-calcular rho_reduced_0 si falló antes, o usarlo si existe
                if 'entanglement_entropy' in metrics: # Si la entropía se calculó, rho_reduced_0 existe
                    # Calcular pureza
                    purity = np.real(np.trace(np.dot(rho_reduced_0, rho_reduced_0)))
                    metrics['subsystem_purity'] = purity
                else: # Intentar calcular rho_reduced_0 de nuevo si falló antes
                     rho_reduced_0 = partial_trace(qiskit_sv, qargs=list(range(1, self.num_qubits))).data
                     purity = np.real(np.trace(np.dot(rho_reduced_0, rho_reduced_0)))
                     metrics['subsystem_purity'] = purity
            except Exception as e:
                 logger.warning(f"Error calculando pureza de subsistema 0: {e}")


            # Concurrencia (solo definido para 2 qubits)
            if self.num_qubits == 2:
                 try:
                     # psi = a|00> + b|01> + c|10> + d|11>
                     # Amplitudes del vector de estado
                     a, b, c, d = self._position_space[0], self._position_space[1], self._position_space[2], self._position_space[3]
                     concurrence = 2 * np.abs(a * d - b * c)
                     metrics['concurrence'] = concurrence
                 except Exception as e:
                     logger.warning(f"Error calculando concurrencia: {e}")

        except ImportError:
            # Si alguna función específica de Qiskit falta (poco probable si la importación inicial funcionó)
             raise QiskitNotAvailableError("Funcionalidad requerida de Qiskit no encontrada.")
        except Exception as e:
            # Captura general para errores inesperados en los cálculos
            logger.error(f"Error inesperado calculando métricas de entrelazamiento: {e}")

        return metrics

    def _calculate_momentum_space_internal(self) -> None:
        """Calcula la representación de momentum (FFT normalizada) internamente."""
        if self._position_space is None:
            # Esto no debería ocurrir si se llama desde la propiedad momentum_space
            # cuando position_space existe. Indica un error lógico.
            raise InvalidStateError("Espacio de posición no definido para calcular momentum.")
        try:
            # Usar FFT ortonormalizada (preserva norma L2)
            self._momentum_space = fft(self._position_space, norm="ortho")
            # Verificar norma (opcional, por paranoia numérica)
            # mom_norm = np.linalg.norm(self._momentum_space)
            # if not np.isclose(mom_norm, 1.0, atol=1e-7):
            #     logger.warning(f"Norma del espacio de momentum es {mom_norm:.4f} después de FFT.")
            #     # No renormalizar aquí, la FFT debería ser unitaria.
            logger.debug(f"Espacio de momentum calculado [id={self.id}]")
        except Exception as e:
            logger.error(f"Error en FFT para espacio de momentum: {e}")
            self._momentum_space = None # Asegurar que quede None si falla
            raise QuantumOSError(f"Fallo al calcular FFT para momentum: {e}") from e

    def _calculate_position_space_internal(self) -> None:
        """Calcula la representación de posición (IFFT normalizada desde momentum) internamente."""
        if self._momentum_space is None:
             raise InvalidStateError("Espacio de momentum no definido para calcular posición.")
        try:
            # Usar IFFT ortonormalizada
            new_position_space = ifft(self._momentum_space, norm="ortho")
            # Verificar norma después de IFFT por posibles errores numéricos
            norm = np.linalg.norm(new_position_space)
            if not np.isclose(norm, 1.0, rtol=1e-6, atol=1e-9):
                logger.warning(f"Re-normalizando después de IFFT (norma={norm:.6f})")
                new_position_space /= norm
                # Verificar de nuevo
                if not np.isclose(np.linalg.norm(new_position_space), 1.0, rtol=1e-6, atol=1e-9):
                    raise InvalidStateError("Falló la re-normalización después de IFFT.")

            self._position_space = new_position_space
            logger.debug(f"Espacio de posición recalculado desde momentum [id={self.id}]")
        except Exception as e:
            logger.error(f"Error en IFFT para espacio de posición: {e}")
            self._position_space = None # Podría quedar en estado inconsistente
            raise QuantumOSError(f"Fallo al calcular IFFT para posición: {e}") from e

    # --- Obtención de Probabilidades y Mediciones ---

    def get_probabilities(self, basis: str = "position") -> np.ndarray:
        """
        Obtiene las probabilidades de medir cada estado base en la base especificada.

        Args:
            basis (str): "position" (computacional) o "momentum" (Fourier).

        Returns:
            np.ndarray: Array de probabilidades (tamaño 2^N) que suman ~1.0.

        Raises:
            InvalidStateError: Si el estado base requerido no está disponible.
            ValueError: Si la base no es válida.
        """
        with self._lock:
            probs: Optional[np.ndarray] = None
            if basis == "position":
                if self._position_space is None:
                    raise InvalidStateError("Espacio de posición no cargado.")
                space = self._position_space
            elif basis == "momentum":
                # Acceder via property para asegurar cálculo si es necesario
                mom_space = self.momentum_space
                if mom_space is None:
                    raise InvalidStateError("Espacio de momentum no disponible o no calculable.")
                space = mom_space
            else:
                raise ValueError("Base inválida. Usar 'position' o 'momentum'.")

            # Calcular probabilidades |amplitude|^2
            probs = np.abs(space)**2

            # Asegurar que sumen 1 (por errores de punto flotante)
            sum_probs = np.sum(probs)
            if not np.isclose(sum_probs, 1.0, rtol=1e-6, atol=1e-9):
                 logger.warning(f"Suma de probabilidades en base '{basis}' es {sum_probs:.6f}. Re-normalizando probabilidades.")
                 # Evitar división por cero si la suma es muy pequeña (aunque no debería ocurrir si la norma era 1)
                 if sum_probs > 1e-12:
                     probs /= sum_probs
                 else:
                     # Esto indicaría un estado casi nulo, que no debería pasar la validación inicial
                     logger.error(f"Suma de probabilidades extremadamente baja ({sum_probs}) en base '{basis}'. Estado inválido?")
                     # Podríamos devolver distribución uniforme o lanzar error? Error es más seguro.
                     raise InvalidStateError(f"Suma de probabilidades inválida ({sum_probs}) en base '{basis}'.")

            return probs


    def measure(self, basis: str = "position", num_shots: int = 1024) -> Dict[str, int]:
        """
        Simula mediciones proyectivas en la base especificada.

        Args:
            basis (str): "position" (computacional) o "momentum" (Fourier).
            num_shots (int): Número de mediciones a simular.

        Returns:
            Dict[str, int]: Diccionario de resultados { "bitstring": count }.

        Raises:
            InvalidStateError: Si el estado base requerido no está disponible.
            ValueError: Si num_shots <= 0 o la base es inválida.
        """
        if num_shots <= 0: raise ValueError("num_shots debe ser positivo.")

        with self._lock:
            # Obtener probabilidades validadas (normalizadas)
            try:
                probs = self.get_probabilities(basis)
            except (InvalidStateError, ValueError) as e:
                logger.error(f"No se pueden realizar mediciones: {e}")
                raise

            # Realizar el muestreo multinomial
            indices = np.arange(self.dimension)
            try:
                # np.random.choice es eficiente para esto
                measured_indices = np.random.choice(indices, size=num_shots, p=probs)

                # Contar los resultados
                results: Dict[str, int] = {}
                for idx in measured_indices:
                    # Formatear el índice como un string binario
                    bit_str = format(idx, f'0{self.num_qubits}b')
                    results[bit_str] = results.get(bit_str, 0) + 1

                logger.info(f"Medición simulada (base={basis}) [id={self.id}, shots={num_shots}] completada.")
                return results

            except ValueError as e:
                 # Esto puede ocurrir si las probabilidades no suman exactamente 1 a pesar de la normalización,
                 # o si hay NaNs/Infinitos (lo cual indicaría un error previo).
                 logger.error(f"Error en np.random.choice (probs sum={np.sum(probs)}): {e}")
                 raise InvalidStateError(f"Error en la distribución de probabilidad para muestreo ({basis}): {e}")
            except Exception as e:
                 logger.error(f"Error inesperado durante la simulación de medición ({basis}): {e}")
                 raise QuantumOSError(f"Error en simulación de medición: {e}") from e

    # --- Operaciones Cuánticas (Requieren Qiskit) ---

    def apply_gate(self, gate_name: str, target_qubit: Union[int, List[int]],
                   control_qubit: Optional[Union[int, List[int]]] = None,
                   params: Optional[List[float]] = None) -> None:
        """
        Aplica una compuerta cuántica al estado actual usando simulación ideal
        (Qiskit Aer Statevector simulator).

        Importante: Esta implementación recrea el circuito y el simulador en cada
        llamada, lo cual es correcto pero puede ser ineficiente para secuencias largas.
        No soporta aplicación de ruido con el simulador de vector de estados.

        Args:
            gate_name (str): Nombre de la compuerta Qiskit (e.g., "h", "cx", "rz"). Case-insensitive.
            target_qubit (int o List[int]): Índice(s) del qubit(s) objetivo (0-based).
            control_qubit (int o List[int], optional): Índice(s) del qubit(s) de control.
            params (list[float], optional): Parámetros para compuertas parametrizadas
                                           (e.g., ángulo en radianes para Rz).

        Raises:
            QiskitNotAvailableError: Si Qiskit no está instalado.
            InvalidStateError: Si no hay estado cargado.
            ValueError: Si los parámetros de la compuerta son inválidos (qubits, nombre).
            NotImplementedError: Si se proporciona un `noise_model` (no soportado aquí).
            QuantumOSError: Si ocurre un error durante la simulación de Qiskit.
        """
        if not HAS_QISKIT:
            raise QiskitNotAvailableError("Qiskit es necesario para aplicar compuertas.")
        if self.noise_model is not None:
            raise NotImplementedError("Aplicación de compuertas con noise_model no está soportada "
                                      "con el simulador de vector de estados. Use backend_type "
                                      "DENSITY_MATRIX o QASM y una implementación diferente.")

        with self._lock:
            if self._position_space is None:
                raise InvalidStateError("No hay estado cargado para aplicar compuerta.")

            # --- Validación de Qubits ---
            targets = [target_qubit] if isinstance(target_qubit, int) else list(target_qubit)
            controls = []
            if control_qubit is not None:
                controls = [control_qubit] if isinstance(control_qubit, int) else list(control_qubit)

            all_qubits = targets + controls
            if not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in all_qubits):
                 raise ValueError(f"Índice de qubit inválido o fuera de rango (0 a {self.num_qubits - 1}). Qubits: {all_qubits}")
            if len(set(all_qubits)) != len(all_qubits):
                 raise ValueError(f"Qubits objetivo y de control deben ser únicos. Qubits: {all_qubits}")

            # --- Construcción y Simulación del Circuito ---
            try:
                # Crear circuito temporal para aplicar esta única compuerta al estado actual
                qc = QuantumCircuit(self.num_qubits, name=f"apply_{gate_name}")

                # Inicializar el circuito con el estado actual
                # Usar Statevector de Qiskit para manejar la inicialización
                initial_state = Statevector(self._position_space)
                qc.initialize(initial_state.data, range(self.num_qubits)) # .data da el numpy array

                # Obtener el método de la compuerta del objeto QuantumCircuit
                gate_func_name = gate_name.lower()
                if not hasattr(qc, gate_func_name):
                     raise ValueError(f"Compuerta '{gate_name}' no reconocida por Qiskit QuantumCircuit.")

                gate_method = getattr(qc, gate_func_name)

                # Construir argumentos para el método de la compuerta
                # El orden típico en Qiskit es: params*, control(s), target(s)
                # Pero varía (e.g., cx(control, target), rz(angle, target))
                args_to_pass = []
                if params: args_to_pass.extend(params)

                # Añadir qubits - ¡El orden importa y depende de la compuerta!
                # Casos comunes:
                if gate_func_name in ["cx", "cnot", "cz"]:
                    if len(controls) != 1 or len(targets) != 1:
                        raise ValueError(f"{gate_name.upper()} requiere 1 control y 1 target.")
                    args_to_pass.extend([controls[0], targets[0]])
                elif gate_func_name in ["swap"]:
                     if len(targets) != 2 or controls:
                         raise ValueError("SWAP requiere 2 targets y 0 controles.")
                     args_to_pass.extend(targets)
                elif gate_func_name.startswith("mc"): # Compuertas Multi-Control (e.g., mcx, mcz)
                    # Qiskit espera (control_qubits, target_qubit)
                     if not controls or len(targets) != 1:
                         raise ValueError(f"{gate_name.upper()} requiere controles y 1 target.")
                     args_to_pass.append(controls) # Pasar lista de controles
                     args_to_pass.append(targets[0])
                else: # Compuertas de 1 qubit (con o sin params) o casos no especiales
                    # Asumir que los targets van al final
                     if controls:
                         # Podría ser un C-U gate genérico? La API varía.
                         # Por ahora, lanzar error si hay controles en compuertas no explícitamente manejadas.
                         raise ValueError(f"Compuertas controladas para '{gate_name}' no manejadas genéricamente. Use cx, cz, mcx, etc.")
                     if len(targets) != 1:
                         # Asumir que las compuertas restantes son de 1 qubit
                         raise ValueError(f"Compuerta '{gate_name}' parece ser de 1 qubit, pero se proporcionaron {len(targets)} targets.")
                     args_to_pass.append(targets[0])

                # Aplicar la compuerta
                gate_method(*args_to_pass)

                # --- Simular usando Aer statevector_simulator ---
                # Crear simulador (se podría cachear si se usa repetidamente)
                # Nota: 'statevector_simulator' es un método legacy en Aer. Usar AerSimulator.
                # aer_sim = Aer.get_backend('statevector_simulator') # Legacy
                aer_sim = AerSimulator(method='statevector')

                # Ejecutar la simulación. No se necesita transpile para simulador ideal.
                # 'shots=1' es irrelevante para statevector pero puede ser requerido por execute.
                job = aer_sim.run(qc, shots=1) # Para Qiskit Aer > 0.8
                # job = execute(qc, aer_sim, shots=1) # Para Qiskit < 1.0 con backend
                result = job.result()

                # Obtener el vector de estado resultante
                final_statevector = result.get_statevector(qc)

                # --- Actualizar el estado interno ---
                new_state_data = final_statevector.data # Obtener el numpy array
                # Verificar norma por si acaso (simulador ideal debería ser exacto)
                final_norm = np.linalg.norm(new_state_data)
                if not np.isclose(final_norm, 1.0, rtol=1e-6, atol=1e-9):
                    logger.warning(f"Norma después de aplicar {gate_name.upper()} es {final_norm:.6f}. Re-normalizando.")
                    new_state_data = new_state_data / final_norm

                self._position_space = new_state_data
                self._momentum_space = None # Invalidar caché de momentum
                self._update_metadata_on_change()
                self._dirty = True

                logger.info(f"Compuerta {gate_name.upper()} aplicada [id={self.id}, target={targets}, control={controls}, params={params}]")

            except QiskitNotAvailableError: # Relanzar error específico
                raise
            except ImportError:
                 # Si AerSimulator falla al importar (debería haber sido detectado antes)
                 raise QiskitNotAvailableError("Qiskit Aer backend no disponible.")
            except ValueError as ve: # Errores de validación de compuerta/qubits
                 logger.error(f"Error de valor al aplicar compuerta {gate_name}: {ve}")
                 raise # Relanzar como ValueError
            except Exception as e:
                 # Capturar otros errores de Qiskit o del proceso
                 logger.exception(f"Error inesperado al aplicar compuerta {gate_name} con Qiskit: {e}")
                 # Podríamos intentar restaurar el estado anterior si falló? No trivial.
                 raise QuantumOSError(f"Fallo en simulación Qiskit para {gate_name}: {e}") from e


    def apply_qft(self, inverse: bool = False) -> None:
        """
        Aplica la Transformada Cuántica de Fourier (QFT) o su inversa (IQFT)
        al estado actual usando la relación directa con FFT/IFFT.

        Esto actualiza directamente `_position_space` e invalida `_momentum_space`.

        Args:
            inverse (bool): Si es True, aplica la IQFT en lugar de la QFT.

        Raises:
            InvalidStateError: Si no hay estado cargado.
            QuantumOSError: Si ocurre un error durante el cálculo FFT/IFFT.
        """
        op_name = "IQFT" if inverse else "QFT"
        with self._lock:
            if self._position_space is None:
                raise InvalidStateError(f"No hay estado cargado para aplicar {op_name}.")

            try:
                # QFT |x> --> FFT(|x>)
                # IQFT |x> --> IFFT(|x>)
                # La operación transforma el vector de estado en la base computacional.
                if not inverse:
                    new_position_space = fft(self._position_space, norm="ortho")
                else:
                    new_position_space = ifft(self._position_space, norm="ortho")

                # Verificar norma después de la transformación
                norm = np.linalg.norm(new_position_space)
                if not np.isclose(norm, 1.0, rtol=1e-6, atol=1e-9):
                    logger.warning(f"Re-normalizando después de {op_name} (norma={norm:.6f})")
                    new_position_space /= norm
                    # Doble check
                    if not np.isclose(np.linalg.norm(new_position_space), 1.0, rtol=1e-6, atol=1e-9):
                        raise QuantumOSError(f"Falló la normalización del estado después de {op_name}")

                self._position_space = new_position_space
                self._momentum_space = None # El antiguo momentum ya no es válido
                self._update_metadata_on_change()
                self._dirty = True
                logger.info(f"{op_name} aplicada usando FFT/IFFT [id={self.id}]")

            except Exception as e:
                logger.error(f"Error durante la aplicación de {op_name} vía FFT/IFFT: {e}")
                # Estado puede quedar inconsistente. ¿Marcar como inválido?
                raise QuantumOSError(f"Error aplicando {op_name}") from e

    # --- Visualización (Requiere Matplotlib) ---

    def plot_distribution(self, basis: str = "position", save_plot: bool = True) -> Optional[Figure]:
        """
        Genera y opcionalmente guarda una gráfica de la distribución de probabilidad
        en la base especificada (posición o momentum).

        Args:
            basis (str): "position" o "momentum".
            save_plot (bool): Si es True, guarda la gráfica en el directorio VIS_DIR.

        Returns:
            Optional[Figure]: El objeto Figure de Matplotlib si se generó, o None si Matplotlib no está disponible.
                              La figura NO se cierra automáticamente aquí; el llamador es responsable.

        Raises:
            InvalidStateError: Si el estado no está cargado o la base es inválida.
            ValueError: Si la base no es válida.
            ImportError: Si Matplotlib no está instalado.
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib no disponible, no se puede generar la gráfica.")
            return None

        with self._lock:
            try:
                probs = self.get_probabilities(basis)
            except (InvalidStateError, ValueError) as e:
                logger.error(f"No se puede generar la gráfica: {e}")
                raise

            fig, ax = plt.subplots(figsize=(max(10, self.num_qubits * 0.8), 6)) # Ajustar tamaño
            indices = np.arange(self.dimension)
            # Generar etiquetas de bitstring
            labels = [format(i, f'0{self.num_qubits}b') for i in indices]

            color = 'tab:blue' if basis == "position" else 'tab:orange'
            ax.bar(indices, probs, tick_label=labels, color=color, width=0.8)

            ax.set_xlabel(f'Estado Base ({basis.capitalize()})')
            ax.set_ylabel('Probabilidad')
            ax.set_title(f'Distribución de Probabilidad ({basis.capitalize()}) - N={self.num_qubits} Qubits (ID: {self.id[:8]})')
            ax.set_ylim(bottom=0, top=max(1.0, np.max(probs) * 1.1)) # Ajustar límite superior
            ax.grid(axis='y', linestyle='--', alpha=0.6)

            # Rotar etiquetas si hay muchos qubits para evitar solapamiento
            rotation = 90 if self.num_qubits > 5 else 0
            fontsize = 8 if self.num_qubits > 7 else 10
            ax.tick_params(axis='x', labelrotation=rotation, labelsize=fontsize)
            # Asegurar que todos los ticks se muestren si no son demasiados
            if self.dimension <= 32: # Mostrar todos los ticks para hasta 5 qubits
                 ax.set_xticks(indices)
            else: # Dejar que matplotlib decida para más qubits
                 pass # O usar MaxNLocator

            plt.tight_layout() # Ajustar layout para que no se corten las etiquetas

            if save_plot:
                timestamp = int(time.time())
                filename = f"{basis}_dist_{self.id}_{timestamp}.png"
                filepath = os.path.join(VIS_DIR, filename)
                try:
                    fig.savefig(filepath, dpi=150) # Guardar con buena resolución
                    logger.info(f"Visualización de {basis} guardada: {filepath}")
                except Exception as e:
                    logger.error(f"Error al guardar gráfica de {basis}: {e}")
                    # No relanzar, pero la figura sigue disponible

            return fig # Devolver la figura para posible uso interactivo/display

    # Alias para mantener compatibilidad con llamadas anteriores
    def plot_position_distribution(self, save_plot=True) -> Optional[Figure]:
        return self.plot_distribution("position", save_plot)

    def plot_momentum_distribution(self, save_plot=True) -> Optional[Figure]:
        return self.plot_distribution("momentum", save_plot)

    # --- Persistencia (Guardado / Carga Completa) ---

    def get_state_directory(self) -> str:
        """Devuelve la ruta al directorio específico para este estado."""
        return os.path.join(STATE_DIR, self.id)

    def save_state(self, description: Optional[str] = None, update_description: bool = True) -> None:
        """
        Guarda el estado cuántico actual (vector de posición) y sus metadatos en disco.
        Marca el estado como persistente.

        Args:
            description (str, optional): Una descripción para añadir/actualizar en los metadatos.
            update_description (bool): Si False y description!=None, solo usa la descripción
                                       si no había una antes. Si True, siempre actualiza.

        Raises:
            PersistenceError: Si ocurre un error durante el guardado.
            InvalidStateError: Si el estado (vector de posición) no está definido.
        """
        with self._lock:
            if self._position_space is None:
                raise InvalidStateError("No hay estado (vector de posición) para guardar.")

            state_dir = self.get_state_directory()
            os.makedirs(state_dir, exist_ok=True)

            metadata_path = os.path.join(state_dir, self.METADATA_FILENAME)
            state_vector_path = os.path.join(state_dir, self.STATE_VECTOR_FILENAME)

            try:
                # Actualizar metadatos antes de guardar
                self.metadata.is_persistent = True
                if description is not None:
                    if update_description or not self.metadata.description:
                        self.metadata.description = description
                # Asegura que last_modified, tipo, métricas estén actualizados
                self._update_metadata_on_change()

                # Guardar metadatos como JSON
                try:
                    metadata_dict = self.metadata.to_dict()
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata_dict, f, indent=4, ensure_ascii=False)
                except TypeError as json_err:
                     logger.error(f"Error al serializar metadatos a JSON: {json_err}. Metadata: {self.metadata}")
                     raise PersistenceError(f"Error de serialización JSON: {json_err}") from json_err
                except IOError as io_err:
                     logger.error(f"Error de I/O al escribir metadatos: {io_err}")
                     raise PersistenceError(f"Error de I/O en metadatos: {io_err}") from io_err


                # Guardar el vector de estado (posición) usando numpy
                # Usar np.round para controlar precisión y potencialmente reducir tamaño
                # state_to_save = np.round(self._position_space, decimals=NUMPY_SAVE_PRECISION)
                # O guardar directamente si la precisión es crucial
                state_to_save = self._position_space
                try:
                    np.save(state_vector_path, state_to_save, allow_pickle=False)
                except IOError as io_err:
                     logger.error(f"Error de I/O al escribir vector de estado: {io_err}")
                     raise PersistenceError(f"Error de I/O en vector de estado: {io_err}") from io_err


                self._dirty = False # Marcar como no sucio después de guardar exitosamente
                logger.info(f"Estado guardado exitosamente [id={self.id}, path={state_dir}]")

            except PersistenceError: # Relanzar errores de persistencia ya loggeados
                raise
            except Exception as e:
                logger.exception(f"Error inesperado al guardar estado {self.id}: {e}")
                raise PersistenceError(f"Error inesperado al guardar estado {self.id}: {e}") from e

    @classmethod
    def load_state(cls, state_id: str) -> 'QuantumMomentumRepresentation':
        """
        Carga un estado cuántico previamente guardado desde el disco.

        Args:
            state_id (str): El ID único del estado a cargar.

        Returns:
            QuantumMomentumRepresentation: La instancia del estado cargado.

        Raises:
            PersistenceError: Si el estado no se encuentra o hay un error al leer archivos.
            InvalidStateError: Si los datos cargados son inválidos o inconsistentes.
        """
        state_dir = os.path.join(STATE_DIR, state_id)
        metadata_path = os.path.join(state_dir, cls.METADATA_FILENAME)
        state_vector_path = os.path.join(state_dir, cls.STATE_VECTOR_FILENAME)

        if not os.path.isdir(state_dir):
            raise PersistenceError(f"Directorio de estado no encontrado: {state_dir}")
        if not os.path.exists(metadata_path):
            raise PersistenceError(f"Archivo de metadatos no encontrado: {metadata_path}")
        if not os.path.exists(state_vector_path):
            raise PersistenceError(f"Archivo de vector de estado no encontrado: {state_vector_path}")

        logger.info(f"Intentando cargar estado [id={state_id}] desde {state_dir}")
        try:
            # 1. Cargar metadatos
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            metadata = QuantumStateMetadata.from_dict(metadata_dict)

            # Verificar consistencia del ID cargado
            if metadata.id != state_id:
                logger.warning(f"Inconsistencia de ID: Solicitado '{state_id}', metadata dice '{metadata.id}'. Usando ID de metadata.")
                # ¿Debería lanzar un error aquí? Por ahora, continuamos con el ID del archivo.
                state_id = metadata.id # Usar el ID del archivo como fuente de verdad

            # 2. Cargar vector de estado
            position_space = np.load(state_vector_path, allow_pickle=False)

            # --- Crear nueva instancia y poblarla ---
            # Usar num_qubits de los metadatos cargados
            instance = cls(num_qubits=metadata.num_qubits, id=metadata.id)
            instance.metadata = metadata # Sobrescribir metadata por defecto con la cargada

            # 3. Validar dimensiones y tipo del vector cargado
            expected_dim = 2**instance.num_qubits
            if position_space.shape != (expected_dim,):
                raise InvalidStateError(f"Dimensiones del vector cargado {position_space.shape} "
                                        f"no coinciden con num_qubits={instance.num_qubits} "
                                        f"(esperado: {(expected_dim,)}).")
            if not np.issubdtype(position_space.dtype, np.complexfloating):
                 logger.warning("Vector de estado cargado no es complejo. Convirtiendo.")
                 position_space = position_space.astype(complex)

            # 4. Validar y normalizar el estado cargado
            norm = np.linalg.norm(position_space)
            if np.isclose(norm, 0.0, atol=1e-9):
                raise InvalidStateError("Vector de estado cargado tiene norma cero.")
            if not np.isclose(norm, 1.0, rtol=1e-6, atol=1e-9):
                logger.warning(f"Vector de estado cargado no estaba normalizado (norma={norm:.6f}). Normalizando.")
                position_space = position_space / norm
                if not np.isclose(np.linalg.norm(position_space), 1.0, rtol=1e-6, atol=1e-9):
                     raise InvalidStateError("Falló la normalización del vector de estado cargado.")

            # 5. Asignar estado y marcar como limpio
            instance._position_space = position_space
            instance._momentum_space = None # Forzar recálculo al acceder
            instance.metadata.is_persistent = True # Asegurar que esté marcado como persistente
            instance._dirty = False # Estado recién cargado no está sucio

            # Recalcular tipo/métricas basado en el estado cargado
            instance._update_state_type_and_metrics()

            logger.info(f"Estado cargado y validado exitosamente [id={instance.id}]")
            return instance

        except (IOError, json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Error al leer o parsear archivos para estado {state_id} desde {state_dir}: {e}")
            raise PersistenceError(f"No se pudo cargar el estado {state_id}: {e}") from e
        except InvalidStateError as e: # Capturar errores de validación
             logger.error(f"Datos inválidos al cargar estado {state_id}: {e}")
             raise # Relanzar InvalidStateError
        except Exception as e:
            logger.exception(f"Error inesperado al cargar estado {state_id}: {e}")
            raise PersistenceError(f"Error inesperado al cargar estado {state_id}: {e}") from e

    # --- Checkpointing (Guardado Intermedio Rápido) ---

    def _get_checkpoint_path(self) -> str:
         """Obtiene la ruta completa para el archivo de checkpoint."""
         return os.path.join(CHECKPOINT_DIR, f"{self.CHECKPOINT_PREFIX}{self.id}.npz")

    def _save_checkpoint(self) -> bool:
        """
        Guarda un checkpoint del estado si está marcado como 'dirty' y es persistente.
        Utiliza formato npz (comprimido) para eficiencia.

        Returns:
            bool: True si se guardó un checkpoint, False en caso contrario.
        """
        with self._lock:
            # No guardar si no hay cambios, no hay estado, o el estado no está destinado a ser persistente
            if not self._dirty or self._position_space is None or not self.metadata.is_persistent:
                return False

            checkpoint_path = self._get_checkpoint_path()
            temp_checkpoint_path = checkpoint_path + f".tmp_{uuid.uuid4()}" # Temporal único

            logger.debug(f"Intentando guardar checkpoint para [id={self.id}] en {checkpoint_path}")
            try:
                # Crear diccionario de metadatos para guardar (solo lo esencial?)
                # Por ahora, guardamos toda la metadata.
                metadata_dict = self.metadata.to_dict()

                # Guardar estado y metadatos en un archivo .npz (comprimido)
                np.savez_compressed(temp_checkpoint_path,
                                    position_space=self._position_space,
                                    metadata_json=json.dumps(metadata_dict)) # Guardar metadata como string JSON

                # Renombrar atómicamente (si es posible en el OS) para evitar corrupción
                os.replace(temp_checkpoint_path, checkpoint_path)

                # Actualizar tiempo de último checkpoint y marcar como limpio
                self.metadata.last_checkpoint_time = time.time()
                self._dirty = False
                logger.info(f"Checkpoint guardado exitosamente para [id={self.id}]")
                return True

            except Exception as e:
                logger.error(f"Error al guardar checkpoint para {self.id}: {e}", exc_info=False) # No mostrar traceback completo por defecto
                # Limpiar archivo temporal si existe y falló el renombrado/guardado
                if os.path.exists(temp_checkpoint_path):
                    try:
                        os.remove(temp_checkpoint_path)
                        logger.debug(f"Archivo temporal de checkpoint eliminado: {temp_checkpoint_path}")
                    except OSError as remove_err:
                         logger.error(f"Error al eliminar archivo temporal de checkpoint {temp_checkpoint_path}: {remove_err}")
                return False # Indicar fallo

    def _load_from_checkpoint(self) -> bool:
        """
        Intenta cargar (restaurar) el estado desde el último checkpoint si existe.
        Sobrescribe el estado en memoria si el checkpoint es válido.

        Returns:
            bool: True si se restauró desde checkpoint, False si no.
        """
        checkpoint_path = self._get_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            logger.debug(f"No se encontró checkpoint para {self.id} en {checkpoint_path}")
            return False

        logger.info(f"Intentando restaurar estado [id={self.id}] desde checkpoint: {checkpoint_path}")
        with self._lock:
            try:
                # Cargar datos desde el archivo npz
                # allow_pickle=False es más seguro si los archivos pudieran ser manipulados
                data = np.load(checkpoint_path, allow_pickle=False)

                # Extraer y validar datos
                if 'position_space' not in data or 'metadata_json' not in data:
                    logger.error(f"Checkpoint inválido para {self.id}: Faltan claves 'position_space' o 'metadata_json'.")
                    return False

                position_space_chk = data['position_space']
                # .item() para obtener el string del array 0-d de numpy
                metadata_json_chk = data['metadata_json'].item()
                metadata_dict_chk = json.loads(metadata_json_chk)
                metadata_chk = QuantumStateMetadata.from_dict(metadata_dict_chk)

                # --- Validaciones de consistencia ---
                if metadata_chk.id != self.id:
                    logger.error(f"Inconsistencia de ID en checkpoint {self.id}. Metadata dice '{metadata_chk.id}'. No se cargará.")
                    return False
                if metadata_chk.num_qubits != self.num_qubits:
                    logger.error(f"Inconsistencia de num_qubits en checkpoint {self.id}. "
                                 f"Metadata dice {metadata_chk.num_qubits}, objeto tiene {self.num_qubits}. No se cargará.")
                    return False

                expected_dim = 2**self.num_qubits
                if position_space_chk.shape != (expected_dim,):
                     logger.error(f"Dimensiones inválidas en checkpoint {self.id}: {position_space_chk.shape}, esperado {(expected_dim,)}. No se cargará.")
                     return False
                if not np.issubdtype(position_space_chk.dtype, np.complexfloating):
                     logger.warning(f"Tipo de dato no complejo ({position_space_chk.dtype}) en checkpoint {self.id}. Convirtiendo.")
                     position_space_chk = position_space_chk.astype(complex)

                # --- Restaurar estado y metadata ---
                self._position_space = position_space_chk
                self.metadata = metadata_chk # Reemplazar metadata actual con la del checkpoint
                self._momentum_space = None # Recalcular si es necesario
                self._dirty = False # Estado recién cargado del checkpoint está limpio

                # No es necesario normalizar aquí si asumimos que _save_checkpoint guardó un estado válido.
                # Pero una verificación rápida no hace daño:
                norm_chk = np.linalg.norm(self._position_space)
                if not np.isclose(norm_chk, 1.0, rtol=1e-6, atol=1e-9):
                     logger.warning(f"Estado cargado desde checkpoint {self.id} no está normalizado (norma={norm_chk:.6f}). Intentando normalizar.")
                     self._position_space /= norm_chk
                     if not np.isclose(np.linalg.norm(self._position_space), 1.0, rtol=1e-6, atol=1e-9):
                          logger.error(f"Falló la normalización del estado restaurado desde checkpoint {self.id}. Estado puede ser inválido.")
                          # ¿Qué hacer aquí? ¿Devolver False? Por ahora, dejamos el estado normalizado lo mejor posible.

                logger.info(f"Estado restaurado exitosamente desde checkpoint [id={self.id}]")
                return True

            except FileNotFoundError:
                 logger.warning(f"Checkpoint no encontrado (posible race condition?): {checkpoint_path}")
                 return False
            except (IOError, json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                logger.error(f"Error al leer o parsear checkpoint {self.id}: {e}")
                return False
            except Exception as e:
                logger.exception(f"Error inesperado al cargar desde checkpoint {self.id}: {e}")
                return False

    # --- Gestión del Hilo de Checkpointing ---

    def _checkpoint_worker(self):
        """Función ejecutada por el hilo de checkpointing en segundo plano."""
        logger.info(f"Hilo de checkpoint iniciado para estado [id={self.id}] (Intervalo: {CHECKPOINT_INTERVAL}s)")
        while not self._checkpoint_stop_event.wait(timeout=CHECKPOINT_INTERVAL):
            # El wait() devuelve True si el evento se setea, False si se agota el timeout
            # El bucle continúa mientras el evento NO esté seteado (wait devuelve False)
            try:
                # Solo hacer checkpoint si el estado está marcado como persistente
                # (evita checkpoints para estados temporales)
                if self.metadata.is_persistent:
                    saved = self._save_checkpoint()
                    if saved:
                         logger.debug(f"Checkpoint periódico realizado para {self.id}")
                    # else: # No es necesario loggear si no se guardó (porque no estaba dirty o no era persistente)
                    #    logger.debug(f"Checkpoint periódico omitido para {self.id} (not dirty or not persistent)")

            except Exception as e:
                # Capturar cualquier excepción inesperada DENTRO del bucle del worker
                # para evitar que el hilo muera silenciosamente.
                logger.error(f"Error fatal en ciclo del hilo de checkpoint para {self.id}: {e}", exc_info=True)
                # Esperar un poco antes de reintentar para evitar un bucle cerrado si el error persiste
                time.sleep(60) # Esperar 1 minuto antes del siguiente intento

        # El bucle termina cuando self._checkpoint_stop_event es seteado
        logger.info(f"Hilo de checkpoint terminando para estado [id={self.id}]")

    def start_checkpointing(self, interval_override_sec: Optional[float] = None):
        """
        Inicia el proceso de checkpointing automático en segundo plano si no está activo.
        El estado debe estar marcado como `is_persistent=True` (normalmente al guardar por primera vez)
        para que los checkpoints se guarden efectivamente.

        Args:
            interval_override_sec (Optional[float]): Permite usar un intervalo diferente
                                                    al global CHECKPOINT_INTERVAL para esta instancia.
        """
        effective_interval = interval_override_sec if interval_override_sec is not None else CHECKPOINT_INTERVAL

        if effective_interval <= 0:
             logger.warning(f"Checkpointing deshabilitado para {self.id} (intervalo <= 0).")
             # Asegurarse de detener cualquier hilo existente si se deshabilita
             self.stop_checkpointing(save_final=False)
             return

        # Usar el lock para la verificación y creación del hilo de forma segura
        with self._lock:
            if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
                logger.warning(f"Intento de iniciar checkpointing para {self.id} cuando ya está activo.")
                return

            # Asegurar que el evento de parada esté limpio antes de empezar
            self._checkpoint_stop_event.clear()

            # Crear y empezar el hilo del worker
            self._checkpoint_thread = threading.Thread(
                target=self._checkpoint_worker,
                name=f"CheckpointWorker-{self.id[:8]}",
                daemon=True # Permite que el programa principal termine aunque este hilo siga corriendo
            )
            self._checkpoint_thread.start()
            logger.info(f"Checkpointing iniciado para {self.id} con intervalo {effective_interval}s.")

    def stop_checkpointing(self, save_final: bool = True, timeout_sec: float = 5.0):
        """
        Detiene el hilo de checkpointing si está activo.

        Args:
            save_final (bool): Si True, intenta guardar un último checkpoint antes de detenerse,
                               solo si el estado es persistente y está 'dirty'.
            timeout_sec (float): Tiempo máximo de espera para que el hilo termine limpiamente.
        """
        thread_to_stop: Optional[threading.Thread] = None
        with self._lock:
             if self._checkpoint_thread and self._checkpoint_thread.is_alive():
                  thread_to_stop = self._checkpoint_thread
                  self._checkpoint_stop_event.set() # Señalizar al hilo que se detenga
                  self._checkpoint_thread = None # Marcar como no activo inmediatamente
                  logger.info(f"Señal de parada enviada al hilo de checkpoint para [id={self.id}]...")
             else:
                  logger.debug(f"Checkpointing no estaba activo o ya detenido para [id={self.id}]")

        # Esperar fuera del lock para no bloquear otras operaciones mientras se espera
        if thread_to_stop:
            thread_to_stop.join(timeout=timeout_sec)
            if thread_to_stop.is_alive():
                 logger.warning(f"El hilo de checkpoint para {self.id} no terminó en {timeout_sec}s.")
            else:
                 logger.info(f"Hilo de checkpoint detenido limpiamente para [id={self.id}]")

        # Intentar guardar un checkpoint final fuera del lock y después de intentar detener el hilo
        if save_final:
             logger.debug(f"Intentando guardar checkpoint final para [id={self.id}]...")
             # No necesitamos el lock aquí porque _save_checkpoint ya lo adquiere
             saved = self._save_checkpoint()
             if saved:
                 logger.info(f"Checkpoint final guardado para [id={self.id}]")
             # else: logger.debug(f"No se guardó checkpoint final para [id={self.id}] (no dirty o no persistent)")


    def __del__(self):
        """
        Destructor del objeto. Intenta detener el hilo de checkpointing.
        Nota: El comportamiento de __del__ puede ser impredecible, especialmente
        durante el cierre del intérprete. Es más robusto llamar explícitamente
        a stop_checkpointing() cuando ya no se necesite el objeto.
        """
        try:
            if self._checkpoint_thread and self._checkpoint_thread.is_alive():
                print(f"WARN: QuantumMomentumRepresentation [id={self.id}] siendo destruido "
                      "mientras el hilo de checkpointing aún está activo. Intentando detener...", file=sys.stderr)
                # Llamar a stop con save_final=False para evitar I/O en __del__ si es posible
                # y con un timeout corto.
                self.stop_checkpointing(save_final=False, timeout_sec=1.0)
        except Exception as e:
            # El logger puede no estar disponible aquí. Usar print a stderr.
            print(f"WARN: Error al detener checkpointing durante __del__ para {self.id}: {e}", file=sys.stderr)


# --- Ejemplo de Uso Mejorado (Fuera de la clase) ---
if __name__ == "__main__":
    logger.info("--- Iniciando Ejemplo QuantumOS Knob ---")
    state_instance: Optional[QuantumMomentumRepresentation] = None # Para manejo en finally

    try:
        # 1. Crear una instancia (3 qubits, estado inicial |000>)
        logger.info("1. Creando estado inicial...")
        state_instance = QuantumMomentumRepresentation(num_qubits=3)
        logger.info(f"Estado creado: {state_instance}")
        print(f"   Posición inicial: {np.round(state_instance.position_space, 3)}")

        # 2. Cargar un estado de superposición (GHZ)
        logger.info("\n2. Cargando estado GHZ...")
        ghz_state_dict = {"000": 1/np.sqrt(2), "111": 1/np.sqrt(2)}
        state_instance.load_superposition_state(ghz_state_dict)
        logger.info(f"Estado GHZ cargado: {state_instance}")
        print(f"   Posición GHZ: {np.round(state_instance.position_space, 3)}")
        print(f"   Tipo: {state_instance.metadata.state_type}, Métricas: {state_instance.metadata.entanglement_metrics}")

        # 3. Calcular y mostrar probabilidades
        logger.info("\n3. Calculando probabilidades...")
        pos_probs = state_instance.get_probabilities("position")
        mom_probs = state_instance.get_probabilities("momentum")
        print(f"   Probabilidades Posición: {np.round(pos_probs, 3)}")
        print(f"   Probabilidades Momentum: {np.round(mom_probs, 3)}")

        # 4. Aplicar compuertas (si Qiskit está disponible)
        logger.info("\n4. Aplicando compuertas (si Qiskit está disponible)...")
        if HAS_QISKIT:
            try:
                print("   Aplicando Hadamard al qubit 0...")
                state_instance.apply_gate("h", target_qubit=0)
                print(f"   Estado tras H(0): {np.round(state_instance.position_space, 3)}")
                print(f"   Tipo: {state_instance.metadata.state_type}, Métricas: {state_instance.metadata.entanglement_metrics}")

                print("   Aplicando CNOT(0, 1)...")
                state_instance.apply_gate("cx", target_qubit=1, control_qubit=0)
                print(f"   Estado tras CNOT(0,1): {np.round(state_instance.position_space, 3)}")
                print(f"   Tipo: {state_instance.metadata.state_type}, Métricas: {state_instance.metadata.entanglement_metrics}")

                print("   Aplicando Rz(pi/2) al qubit 2...")
                state_instance.apply_gate("rz", target_qubit=2, params=[np.pi/2])
                print(f"   Estado tras Rz(2): {np.round(state_instance.position_space, 3)}")

            except (QiskitNotAvailableError, InvalidStateError, ValueError, NotImplementedError, QuantumOSError) as e:
                logger.error(f"Error al aplicar compuerta: {e}")
        else:
            logger.warning("Qiskit no encontrado, saltando aplicación de compuertas.")

        # 5. Realizar mediciones simuladas
        logger.info("\n5. Realizando mediciones simuladas...")
        pos_measurement = state_instance.measure("position", num_shots=2048)
        mom_measurement = state_instance.measure("momentum", num_shots=2048)
        print(f"   Medición (posición, 2048 shots): {pos_measurement}")
        print(f"   Medición (momentum, 2048 shots): {mom_measurement}")

        # 6. Generar y guardar gráficas (si Matplotlib está disponible)
        logger.info("\n6. Generando gráficas (si Matplotlib está disponible)...")
        if HAS_MATPLOTLIB:
            try:
                fig_pos = state_instance.plot_position_distribution(save_plot=True)
                if fig_pos: plt.close(fig_pos) # Cerrar figura para liberar memoria
                fig_mom = state_instance.plot_momentum_distribution(save_plot=True)
                if fig_mom: plt.close(fig_mom)
            except Exception as e:
                logger.error(f"Error al generar/guardar plots: {e}")
        else:
            logger.warning("Matplotlib no encontrado, saltando generación de gráficas.")

        # 7. Persistencia
        logger.info("\n7. Probando persistencia (Guardar y Cargar)...")
        state_id = state_instance.id
        current_desc = "Estado de prueba después de H, CNOT, Rz"
        try:
            print(f"   Guardando estado {state_id}...")
            state_instance.save_state(description=current_desc)
            print(f"   Estado guardado. Marcado como persistente: {state_instance.metadata.is_persistent}")

            # Guardar referencia al estado actual antes de borrar
            original_state_vector = state_instance.position_space.copy()

            # Simular cierre y reapertura: borrar instancia actual
            print("   Eliminando instancia actual de memoria...")
            del state_instance
            state_instance = None # Asegurar que no hay referencia

            # Cargar el estado guardado
            print(f"   Cargando estado {state_id} desde disco...")
            # Restaurar en la misma variable
            state_instance = QuantumMomentumRepresentation.load_state(state_id)
            print(f"   Estado cargado exitosamente.")
            print(f"   Descripción cargada: '{state_instance.metadata.description}'")
            print(f"   Vector de posición cargado (primeros 4): {np.round(state_instance.position_space[:4], 3)}")

            # Verificar que el estado cargado es igual al original
            if np.allclose(original_state_vector, state_instance.position_space):
                print("   VERIFICACIÓN: Estado cargado coincide con el original.")
            else:
                print("   ERROR: Estado cargado NO coincide con el original.")

            # Verificar que el momentum se puede recalcular
            loaded_mom_probs = state_instance.get_probabilities("momentum")
            print(f"   Probabilidades Momentum (del estado cargado): {np.round(loaded_mom_probs, 3)}")

        except (PersistenceError, InvalidStateError) as e:
            logger.error(f"Error durante la persistencia: {e}")
            # Si falla la carga, state_instance puede ser None

        # 8. Checkpointing (solo si la carga fue exitosa)
        logger.info("\n8. Probando Checkpointing...")
        if state_instance:
             # Asegurarse de que sea persistente para que el checkpointing funcione
             if not state_instance.metadata.is_persistent:
                 logger.warning("El estado cargado no está marcado como persistente, el checkpointing no guardará.")
                 # Forzarlo para el ejemplo (o llamar save_state de nuevo)
                 state_instance.metadata.is_persistent = True

             print(f"   Iniciando checkpointing (Intervalo: {CHECKPOINT_INTERVAL}s)...")
             state_instance.start_checkpointing()
             # El estado está 'limpio' después de cargar, el primer checkpoint no ocurrirá hasta que cambie

             try:
                 # Simular trabajo y cambios para ensuciar el estado
                 print("   Simulando trabajo (espera 3s)...")
                 time.sleep(3)
                 if HAS_QISKIT:
                     try:
                         state_instance.apply_gate("x", 0)
                         print(f"   Compuerta X aplicada a qubit 0. Estado ahora 'dirty': {state_instance._dirty}")
                     except Exception as e:
                          logger.error(f"Error aplicando X(0) en demo de checkpoint: {e}")
                 else:
                     # Simular cambio manual si no hay Qiskit
                     with state_instance._lock:
                          if state_instance._position_space is not None:
                             state_instance._position_space[0] += 0.001 # Pequeño cambio para ensuciar
                             state_instance._position_space /= np.linalg.norm(state_instance._position_space)
                             state_instance._dirty = True
                             print(f"   Estado modificado manualmente. Estado ahora 'dirty': {state_instance._dirty}")


                 # Esperar un tiempo corto para ver si ocurre un checkpoint (si el intervalo es muy corto)
                 # O simplemente continuar y detenerlo.
                 wait_time = 5 # Esperar 5 segundos más
                 print(f"   Esperando {wait_time}s (checkpoint podría ocurrir si el intervalo es corto)...")
                 time.sleep(wait_time)

                 # Detener checkpointing y guardar un checkpoint final si estaba dirty
                 print("   Deteniendo checkpointing...")
                 state_instance.stop_checkpointing(save_final=True) # save_final=True es el default implícito aquí

                 # Simular un crash y recuperación desde checkpoint
                 print("   Simulando carga desde checkpoint...")
                 # Guardar el estado actual para comparar
                 state_before_restore = state_instance.position_space.copy() if state_instance.position_space is not None else None

                 restored = state_instance._load_from_checkpoint()
                 if restored:
                     print("   Estado restaurado exitosamente desde el último checkpoint.")
                     if state_before_restore is not None and np.allclose(state_before_restore, state_instance.position_space):
                         print("   VERIFICACIÓN: Estado restaurado coincide con el estado justo antes de restaurar (porque se guardó checkpoint final).")
                     elif state_before_restore is not None:
                          print("   INFO: Estado restaurado es diferente al estado justo antes (esperado si no se guardó checkpoint final o hubo más cambios).")

                 else:
                      logger.warning("   No se pudo restaurar desde checkpoint (quizás no existía o hubo error).")

             except Exception as e:
                 logger.error(f"Error durante el ejemplo de checkpointing: {e}")
             # El finally abajo se encargará de detener el checkpointing si algo falla

        else:
            logger.warning("Saltando demo de checkpointing porque la carga del estado falló.")

    except Exception as e:
        logger.exception("Error fatal en el script principal.") # Imprime traceback completo

    finally:
        # --- Limpieza ---
        logger.info("\n--- Limpieza Final ---")
        if 'state_instance' in locals() and state_instance is not None:
             # Detener explícitamente el checkpointing si aún existe la instancia
             print("   Asegurando que el checkpointing esté detenido...")
             try:
                 state_instance.stop_checkpointing(save_final=False) # No guardar al final del script
             except Exception as e:
                 logger.error(f"Error al detener checkpointing en finally: {e}")

        # Cerrar todas las figuras de Matplotlib que pudieran quedar abiertas
        if HAS_MATPLOTLIB:
            try:
                plt.close('all')
                print("   Figuras de Matplotlib cerradas.")
            except Exception:
                pass # Ignorar errores al cerrar plots

        logger.info("--- Ejemplo QuantumOS Knob Finalizado ---")
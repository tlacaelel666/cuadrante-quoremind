# -*- coding: utf-8 -*-
"""
script completo que simula el ciclo de vida de tu sistema híbrido, desde la decisión inteligente basada en métricas hasta la manipulación del qubit, la transducción a fotón, la transmisión, detección y retroalimentación
fecha 06-04-2025
autor Jacobo Tlacaelel Mina Rodríguez 
versión QuoreMind v1.0
"""

import time
import numpy as np
import math
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuoreMindSystem")

# --- Clases de Estado y Métricas ---

class EstadoQubit(Enum):
    """Estados posibles para un qubit superconductor"""
    GROUND = auto()  # |0⟩
    EXCITED = auto()  # |1⟩
    SUPERPOSITION = auto()  # α|0⟩ + β|1⟩
    UNKNOWN = auto()  # Estado desconocido o indeterminado

@dataclass
class EstadoComplejo:
    """Representación de un estado cuántico completo"""
    alpha: complex = complex(1.0) # Amplitud del estado |0⟩
    beta: complex = complex(0.0)   # Amplitud del estado |1⟩

    def __post_init__(self):
        # Normalización automática
        self.normalize()

    def normalize(self):
        """Normaliza el vector de estado."""
        norm_sq = abs(self.alpha)**2 + abs(self.beta)**2
        if norm_sq > 1e-12: # Evitar división por cero o normas muy pequeñas
            norm = np.sqrt(norm_sq)
            self.alpha /= norm
            self.beta /= norm
        else:
            # Si la norma es cero, resetear a |0⟩ por seguridad
            self.alpha = complex(1.0)
            self.beta = complex(0.0)

    @property
    def vector(self) -> np.ndarray:
        """Devuelve el vector de estado como array numpy"""
        return np.array([self.alpha, self.beta], dtype=complex)

    def probabilidad_0(self) -> float:
        """Probabilidad de medir |0⟩"""
        return abs(self.alpha)**2

    def probabilidad_1(self) -> float:
        """Probabilidad de medir |1⟩"""
        return abs(self.beta)**2

    def fase_relativa(self) -> float:
        """Calcula la fase relativa entre beta y alpha en radianes."""
        if abs(self.alpha) < 1e-9 or abs(self.beta) < 1e-9:
            return 0.0 # Fase no bien definida si una amplitud es cero
        return np.angle(self.beta) - np.angle(self.alpha)

    def __str__(self) -> str:
        return f"{self.alpha.real:+.4f}{self.alpha.imag:+.4f}j |0⟩ + {self.beta.real:+.4f}{self.beta.imag:+.4f}j |1⟩ (P0={self.probabilidad_0():.3f})"

@dataclass
class MetricasSistema:
    """Métricas del sistema para la toma de decisiones en QuoreMind"""
    ciclo: int
    tiempo_coherencia: float  # Tiempo de coherencia estimado del qubit en microsegundos
    temperatura: float  # Temperatura del sistema en milikelvin
    senal_ruido: float  # Relación señal-ruido (SNR) de lectura/control
    tasa_error: float  # Tasa de error de bit cuántico (QBER) estimada
    fotones_perdidos_acum: int # Contador *acumulado* de fotones que no llegaron al destino
    calidad_transduccion: float  # Calidad estimada de la transducción (0 a 1)
    estado_enlace: Optional[Dict[str, Any]] = None # Estado del enlace óptico
    voltajes_control: Optional[List[float]] = None # Voltajes de control en varios puntos del sistema

    def __str__(self) -> str:
        return (f"Métricas Ciclo {self.ciclo}: T_coh={self.tiempo_coherencia:.2f}μs, "
                f"Temp={self.temperatura:.2f}mK, SNR={self.senal_ruido:.2f}, "
                f"QBER={self.tasa_error:.4f}, Transd={self.calidad_transduccion:.2f}, "
                f"Fotones Perdidos={self.fotones_perdidos_acum}")

# --- Clases de Componentes Físicos (Simulados) ---

class QubitSuperconductor:
    """Modelo simplificado de un qubit superconductor"""

    def __init__(self, id_qubit: str = "Q0", temp_inicial: float = 15.0, t_coherencia_max: float = 100.0):
        self.id = id_qubit
        self.estado_basico = EstadoQubit.GROUND
        self.estado_complejo = EstadoComplejo(complex(1.0), complex(0.0))
        self.tiempo_ultimo_reset = time.time()
        self.t_coherencia_max_base = t_coherencia_max # microsegundos a temperatura base
        self._temperatura = temp_inicial # milikelvin
        self.actualizar_coherencia_por_temp()
        logger.info(f"Qubit {self.id} inicializado a |0⟩, Temp={self._temperatura:.1f}mK, T_coh_max={self.tiempo_coherencia_max:.1f}μs")

    @property
    def temperatura(self) -> float:
        return self._temperatura

    @temperatura.setter
    def temperatura(self, valor: float):
        temp_anterior = self._temperatura
        self._temperatura = max(10.0, min(valor, 50.0)) # Limitada entre 10mK y 50mK
        if temp_anterior != self._temperatura:
             self.actualizar_coherencia_por_temp()
             logger.debug(f"Qubit {self.id} temp actualizada a {self._temperatura:.1f}mK, T_coh_max={self.tiempo_coherencia_max:.1f}μs")

    def actualizar_coherencia_por_temp(self):
        """Ajusta el tiempo máximo de coherencia basado en la temperatura."""
        # Modelo simple: coherencia disminuye linealmente al aumentar temp sobre 10mK
        factor_temp = max(0, 1.0 - ((self._temperatura - 10.0) / 40.0) * 0.8) # Pierde hasta 80% a 50mK
        self.tiempo_coherencia_max = self.t_coherencia_max_base * factor_temp

    def tiempo_desde_reset(self) -> float:
        """Tiempo transcurrido desde el último reset en segundos"""
        return time.time() - self.tiempo_ultimo_reset

    def aplicar_rotacion(self, eje: str, angulo: float):
        """Aplica una rotación en la esfera de Bloch."""
        cos_medio = math.cos(angulo / 2)
        sin_medio = math.sin(angulo / 2)

        if eje.upper() == 'X':
            matriz_rot = np.array([[cos_medio, -1j * sin_medio], [-1j * sin_medio, cos_medio]], dtype=complex)
        elif eje.upper() == 'Y':
            matriz_rot = np.array([[cos_medio, -sin_medio], [sin_medio, cos_medio]], dtype=complex)
        elif eje.upper() == 'Z':
            matriz_rot = np.array([[np.exp(-1j * angulo/2), 0], [0, np.exp(1j * angulo/2)]], dtype=complex)
        else:
            raise ValueError(f"Eje de rotación desconocido: {eje}")

        # Aplicar la matriz al estado
        vector_actual = self.estado_complejo.vector
        nuevo_vector = np.matmul(matriz_rot, vector_actual)
        self.estado_complejo = EstadoComplejo(nuevo_vector[0], nuevo_vector[1])

        # Actualizar estado básico basado en probabilidades (aproximado)
        prob_0 = self.estado_complejo.probabilidad_0()
        if abs(prob_0 - 1.0) < 0.01: # Margen pequeño
            self.estado_basico = EstadoQubit.GROUND
        elif abs(prob_0 - 0.0) < 0.01:
            self.estado_basico = EstadoQubit.EXCITED
        else:
            self.estado_basico = EstadoQubit.SUPERPOSITION

        #logger.debug(f"Qubit {self.id}: Rotación {eje}({angulo:.4f}) -> {self.estado_complejo}")

    def aplicar_hadamard(self):
        """Aplica la compuerta Hadamard."""
        H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        nuevo_vector = np.matmul(H, self.estado_complejo.vector)
        self.estado_complejo = EstadoComplejo(nuevo_vector[0], nuevo_vector[1])
        self.estado_basico = EstadoQubit.SUPERPOSITION # Hadamard siempre crea superposición (excepto de |+> o |->)
        #logger.debug(f"Qubit {self.id}: Hadamard -> {self.estado_complejo}")

    def aplicar_fase_s(self):
        """Aplica la compuerta de Fase S (sqrt(Z))."""
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        nuevo_vector = np.matmul(S, self.estado_complejo.vector)
        self.estado_complejo = EstadoComplejo(nuevo_vector[0], nuevo_vector[1])
        #logger.debug(f"Qubit {self.id}: Fase S -> {self.estado_complejo}")

    def reset(self):
        """Reinicia el qubit al estado base |0⟩"""
        self.estado_basico = EstadoQubit.GROUND
        self.estado_complejo = EstadoComplejo(complex(1.0), complex(0.0))
        self.tiempo_ultimo_reset = time.time()
        logger.info(f"Qubit {self.id} reiniciado a estado |0⟩")

    def simular_decoherencia(self):
        """Simula la decoherencia del qubit con el tiempo."""
        # Modelo T1 (relajación de amplitud) y T2 (decoherencia de fase)
        t = self.tiempo_desde_reset() * 1e6  # tiempo en microsegundos
        t1 = self.tiempo_coherencia_max * 1.5 # T1 suele ser mayor que T2
        t2 = self.tiempo_coherencia_max       # Usamos T_coh_max como T2

        if t > t2 / 10 : # Aplicar si ha pasado un tiempo significativo
            # Factor de decaimiento de fase (T2)
            factor_fase = np.exp(-t / t2)
            # Factor de decaimiento de amplitud (T1)
            factor_amp = np.exp(-t / t1)

            # Aplicar decaimiento de amplitud a beta (estado excitado)
            beta_amp = abs(self.estado_complejo.beta) * factor_amp
            # Recalcular alpha para mantener norma (aproximado, mejor usar matriz densidad)
            alpha_amp_sq = 1.0 - beta_amp**2
            alpha_amp = np.sqrt(max(0, alpha_amp_sq))

            # Aplicar decaimiento de fase a la fase relativa
            fase_rel_original = self.estado_complejo.fase_relativa()
            # La fase decae, pero modelarlo así es simplista. Mejor afectar coherencias off-diagonal.
            # Simplificación: reducir componente imaginaria (similar a lo anterior)
            alpha_new = alpha_amp * (np.cos(np.angle(self.estado_complejo.alpha)) + 1j*np.sin(np.angle(self.estado_complejo.alpha))*factor_fase)
            beta_new = beta_amp * (np.cos(np.angle(self.estado_complejo.beta)) + 1j*np.sin(np.angle(self.estado_complejo.beta))*factor_fase)

            self.estado_complejo = EstadoComplejo(alpha_new, beta_new)

            # Si la decoherencia es muy alta, colapsar a estado clásico
            if t > t2:
                logger.warning(f"Qubit {self.id} ha decoherido significativamente (t={t:.1f}μs > T2={t2:.1f}μs)")
                self.medir() # Forzar colapso por medición simulada
                self.estado_basico = EstadoQubit.GROUND if abs(self.estado_complejo.alpha) > 0.5 else EstadoQubit.EXCITED


    def medir(self) -> int:
        """Simula una medición en la base computacional Z. Colapsa el estado."""
        prob_0 = self.estado_complejo.probabilidad_0()
        resultado = 0 if np.random.random() < prob_0 else 1
        
        # Colapsar estado
        if resultado == 0:
            self.estado_complejo = EstadoComplejo(complex(1.0), complex(0.0))
            self.estado_basico = EstadoQubit.GROUND
        else:
            self.estado_complejo = EstadoComplejo(complex(0.0), complex(1.0))
            self.estado_basico = EstadoQubit.EXCITED
        
        logger.info(f"Qubit {self.id} medido: Resultado = {resultado}. Estado colapsado a |{resultado}⟩")
        return resultado

    def __str__(self) -> str:
        return f"Qubit[{self.id}|{self.estado_basico.name}]: {self.estado_complejo}"

# --- Clases de Operaciones y Control ---

class OperacionCuantica(Enum):
    """Operaciones cuánticas disponibles"""
    ROTACION_X = auto()
    ROTACION_Y = auto()
    ROTACION_Z = auto()
    HADAMARD = auto()
    FASE_S = auto()
    RESET = auto()
    MEDICION = auto() # Añadida operación de medición explícita

@dataclass
class ParametrosOperacion:
    """Parámetros para una operación cuántica"""
    tipo: OperacionCuantica
    angulo: Optional[float] = None  # Para rotaciones
    # Parámetros de pulso (opcionales, podrían ser calculados por MicrowaveControl)
    duracion_pulso: Optional[float] = None  # Duración en nanosegundos
    amplitud: Optional[float] = None  # Amplitud del pulso
    fase: Optional[float] = None  # Fase del pulso

# --- Clase de Control QuoreMind ---

class QuoreMind:
    """Sistema de decisión y control inteligente."""
    
    def __init__(self):
        self.metricas_historicas: List[MetricasSistema] = []
        self.decisiones_previas: List[ParametrosOperacion] = []
        self.resultados_previos: List[Dict[str, Any]] = []
        self.contador_ciclos = 0
        self.fotones_perdidos_totales = 0
        self.tasa_exito_global = 0.9 # Empezar optimista
        self.estado_sistema = "listo"
        
        # Parámetros de decisión (podrían ser aprendidos)
        self.factores_decision = {
            "peso_coherencia": 0.4,
            "peso_temperatura": 0.1,
            "peso_snr": 0.2,
            "peso_transduccion": 0.3,
            "umbral_calibracion": 0.7, # Umbral para decidir si calibrar
            "factor_exploracion": 0.05 # Probabilidad de probar algo diferente
        }
    
    def obtener_metricas_actuales(self, qubit: QubitSuperconductor, canal: 'OpticalChannel', detector: 'PhotonDetector') -> MetricasSistema:
        """Obtiene y simula las métricas actuales del sistema."""
        self.contador_ciclos += 1
        
        # Simular coherencia actual
        t_coh_max = qubit.tiempo_coherencia_max
        t_desde_reset = qubit.tiempo_desde_reset() * 1e6 # a µs
        # Coherencia disminuye exponencialmente (simplificado)
        coh_actual = t_coh_max * np.exp(- t_desde_reset / t_coh_max)
        coh_actual = max(5.0, coh_actual) # Mínimo de 5µs

        # Simular fluctuaciones de temperatura
        temp_actual = qubit.temperatura + np.random.normal(0, 0.3)
        temp_actual = max(10.0, min(50.0, temp_actual))
        qubit.temperatura = temp_actual # Actualizar temperatura real del qubit (afecta coherencia)

        # Simular calidad de transducción (depende de temp)
        calidad_trans = 0.98 - ((temp_actual - 10.0) / 40.0)**1.5 * 0.5 # No lineal
        calidad_trans = max(0.4, min(0.99, calidad_trans))

        # Simular SNR (depende de calidad trans, coherencia)
        snr_base = 28.0 # dB
        snr_actual = snr_base * calidad_trans * (coh_actual / t_coh_max) + np.random.normal(0, 1.0)
        snr_actual = max(8.0, min(35.0, snr_actual))

        # Simular QBER (depende de SNR, calidad trans, perdidas canal)
        qber = 0.005 + (1.0 - calidad_trans)*0.03 + (35.0 - snr_actual)/50.0 * 0.04 + canal.perdida_acumulada * 0.01
        qber = max(0.0005, min(0.15, qber))

        # Fotones perdidos (actualizar desde el contador global)
        self.fotones_perdidos_totales = canal.fotones_perdidos # Obtener del canal

        metricas = MetricasSistema(
            ciclo=self.contador_ciclos,
            tiempo_coherencia=coh_actual,
            temperatura=temp_actual,
            senal_ruido=snr_actual,
            tasa_error=qber,
            fotones_perdidos_acum=self.fotones_perdidos_totales,
            calidad_transduccion=calidad_trans,
            # Podrían añadirse estado_enlace y voltajes si se simularan con más detalle
        )
        
        self.metricas_historicas.append(metricas)
        #logger.info(f"Métricas Ciclo {self.contador_ciclos}: {metricas}")
        return metricas
    
    def _calcular_cosenos_directores(self, metricas: MetricasSistema) -> Tuple[float, float, float]:
        """Calcula cosenos basado en métricas normalizadas."""
        # Normalizar métricas clave a [0, 1] aprox
        norm_coh = np.clip(metricas.tiempo_coherencia / 100.0, 0, 1) # Max T_coh 100us
        norm_temp_inv = np.clip(1.0 - (metricas.temperatura - 10.0) / 40.0, 0, 1) # Mejor a baja temp
        norm_snr = np.clip((metricas.senal_ruido - 10.0) / 25.0, 0, 1) # Rango 10-35 dB
        norm_trans = np.clip(metricas.calidad_transduccion, 0, 1)

        # Mapeo heurístico a componentes (cómo afectan la "dirección" preferida)
        comp_x = norm_coh * 0.6 + norm_snr * 0.4     # Coherencia y SNR -> Eje X?
        comp_y = norm_trans * 0.7 + norm_snr * 0.3  # Transducción y SNR -> Eje Y?
        comp_z = norm_temp_inv * 0.5 + norm_coh * 0.5 # Temperatura (inversa) y Coherencia -> Eje Z?

        magnitud = math.sqrt(comp_x**2 + comp_y**2 + comp_z**2)
        if magnitud < 1e-9: return (1.0, 0.0, 0.0) # Evitar división por cero, devolver eje X por defecto
        
        return (comp_x / magnitud, comp_y / magnitud, comp_z / magnitud)

    def calcular_angulo_control(self, metricas: MetricasSistema) -> float:
        """Calcula ángulo de rotación basado en métricas."""
        base_angle = math.pi / 2 # Rotación de 90 grados como base?

        # Factores de ajuste basados en calidad
        factor_coh = np.clip(metricas.tiempo_coherencia / 50.0, 0.5, 1.2) # Más coherencia -> más rotación?
        factor_qual = np.clip(metricas.calidad_transduccion, 0.5, 1.1)

        angulo = base_angle * factor_coh * factor_qual

        # Añadir exploración
        if np.random.random() < self.factores_decision["factor_exploracion"]:
            perturbacion = np.random.normal(0, math.pi / 16) # Perturbar +/- ~11 grados
            angulo += perturbacion
            #logger.debug(f"Aplicando perturbación ángulo: {math.degrees(perturbacion):.1f}°")

        angulo = np.clip(angulo, 0, math.pi) # Limitar a [0, pi]
        #logger.debug(f"Ángulo de control calculado: {math.degrees(angulo):.1f}°")
        return angulo

    def decidir_operacion(self, metricas: MetricasSistema) -> ParametrosOperacion:
        """Decide la operación cuántica."""
        # Evaluar necesidad de calibración/reset
        calidad_general = (
            self.factores_decision["peso_coherencia"] * np.clip(metricas.tiempo_coherencia / 80.0, 0, 1) +
            self.factores_decision["peso_snr"] * np.clip(metricas.senal_ruido / 30.0, 0, 1) +
            self.factores_decision["peso_transduccion"] * metricas.calidad_transduccion
        ) / (self.factores_decision["peso_coherencia"] + self.factores_decision["peso_snr"] + self.factores_decision["peso_transduccion"])
        
        if calidad_general < self.factores_decision["umbral_calibracion"]:
             logger.warning(f"Calidad general baja ({calidad_general:.3f}), considerando RESET.")
             # Decidir si resetear o intentar otra operación
             if np.random.random() < 0.8: # Alta probabilidad de reset si la calidad es baja
                  self.estado_sistema = "calibrando"
                  return ParametrosOperacion(tipo=OperacionCuantica.RESET)

        # Si no se resetea, elegir operación de rotación/Hadamard/Fase
        self.estado_sistema = "operando"
        cos_x, cos_y, cos_z = self._calcular_cosenos_directores(metricas)
        angulo = self.calcular_angulo_control(metricas)

        # Seleccionar eje basado en cosenos
        ejes = {'X': abs(cos_x), 'Y': abs(cos_y), 'Z': abs(cos_z)}
        eje_elegido = max(ejes, key=ejes.get)
        
        # Asignar tipo de operación
        if eje_elegido == 'X': operacion_tipo = OperacionCuantica.ROTACION_X
        elif eje_elegido == 'Y': operacion_tipo = OperacionCuantica.ROTACION_Y
        else: operacion_tipo = OperacionCuantica.ROTACION_Z

        # Ocasionalmente probar Hadamard o Fase S si la calidad es buena
        if calidad_general > 0.85 and np.random.random() < 0.15:
             operacion_tipo = np.random.choice([OperacionCuantica.HADAMARD, OperacionCuantica.FASE_S])
             angulo = None # No necesitan ángulo explícito aquí

        # Definir parámetros (duración/amplitud podrían definirse aquí o en MicrowaveControl)
        params = ParametrosOperacion(tipo=operacion_tipo, angulo=angulo)
        self.decisiones_previas.append(params)
        op_name = operacion_tipo.name
        angle_deg = f"{math.degrees(angulo):.1f}°" if angulo is not None else "N/A"
        #logger.info(f"Decisión Ciclo {self.contador_ciclos}: Operación={op_name}, Ángulo={angle_deg}")
        return params

    def actualizar_aprendizaje(self, resultado_ciclo: Dict[str, Any]):
        """Actualiza parámetros basado en el éxito del ciclo."""
        self.resultados_previos.append(resultado_ciclo)
        
        # Usar 'exito_deteccion' y 'error_medicion' para evaluar el éxito
        exito = resultado_ciclo.get("exito_deteccion", False) and not resultado_ciclo.get("error_medicion", True)
        
        # Calcular tasa de éxito reciente (últimos 20 ciclos)
        historial = min(len(self.resultados_previos), 20)
        exitos_recientes = sum(1 for r in self.resultados_previos[-historial:]
                              if r.get("exito_deteccion", False) and not r.get("error_medicion", True))
        tasa_exito_reciente = exitos_recientes / historial if historial > 0 else 0.0

        # Actualizar tasa global
        self.tasa_exito_global = 0.95 * self.tasa_exito_global + 0.05 * tasa_exito_reciente
        
        # Ajustar exploración (más simple)
        if tasa_exito_reciente < 0.6 and len(self.resultados_previos) > 10:
             self.factores_decision["factor_exploracion"] = min(0.3, self.factores_decision["factor_exploracion"] + 0.01)
        elif tasa_exito_reciente > 0.85:
             self.factores_decision["factor_exploracion"] = max(0.02, self.factores_decision["factor_exploracion"] - 0.005)

        #logger.debug(f"Aprendizaje actualizado: Tasa éxito global={self.tasa_exito_global:.3f}, Exploración={self.factores_decision['factor_exploracion']:.3f}")

    def registrar_evento(self, evento: str):
        logger.info(f"EVENTO Ciclo {self.contador_ciclos}: {evento}")

    def registrar_error(self, error: str):
        logger.error(f"ERROR Ciclo {self.contador_ciclos}: {error}")

    def obtener_estado_sistema(self) -> Dict[str, Any]:
        """Devuelve resumen del estado."""
        return {
            "ciclo": self.contador_ciclos,
            "tasa_exito_global": self.tasa_exito_global,
            "factor_exploracion": self.factores_decision["factor_exploracion"],
            "estado_operativo": self.estado_sistema,
            "ultima_metrica": str(self.metricas_historicas[-1]) if self.metricas_historicas else "N/A",
            "ultima_decision": self.decisiones_previas[-1].tipo.name if self.decisiones_previas else "N/A"
        }

# --- Clase de Control de Microondas ---

class MicrowaveControl:
    """Controlador simulado de pulsos de microondas."""
    
    def __init__(self):
        self.frecuencia_base = 5.1 # GHz
        self.precision_tiempo = 0.1 # ns
        self.latencia_aplicacion = 5 # ns de retraso simulado
        # Calibración podría ser más compleja
        self.calibracion = {"offset_frecuencia": 0.0, "factor_amplitud": 1.0, "offset_fase": 0.0}
        logger.info("MicrowaveControl inicializado.")

    def traducir_operacion_a_pulso(self, operacion: ParametrosOperacion) -> Dict[str, Any]:
        """Traduce operación lógica a parámetros de pulso físico (simulado)."""
        # Parámetros base
        duracion_base_ns = 15.0
        amplitud_base = 0.95
        
        params = {
            "tipo_operacion": operacion.tipo.name,
            "angulo_logico": operacion.angulo,
            "duracion": duracion_base_ns,
            "amplitud": amplitud_base,
            "frecuencia": self.frecuencia_base + self.calibracion["offset_frecuencia"],
            "fase": self.calibracion["offset_fase"],
            "forma": "gaussiana_derivada" # DRAG pulses etc.
        }

        # Ajustes específicos por operación
        if operacion.tipo == OperacionCuantica.ROTACION_X:
             # Rotación X requiere pulso en cuadratura (fase 0 o pi?)
             params["fase"] += 0.0
             # La duración/amplitud determina el ángulo de rotación
             # Aquí asumimos que MicrowaveControl sabe cómo lograr 'angulo'
        elif operacion.tipo == OperacionCuantica.ROTACION_Y:
             params["fase"] += math.pi / 2
        elif operacion.tipo == OperacionCuantica.ROTACION_Z:
             # Virtual: se aplica cambiando marco de referencia (fase de futuros pulsos)
             params["virtual"] = True
             params["amplitud"] = 0.0 # No hay pulso físico
        elif operacion.tipo == OperacionCuantica.HADAMARD:
             # Secuencia específica o pulso calibrado
             params["tipo_operacion"] = "HADAMARD_PULSE" # Pulso especial
             params["duracion"] = 25.0 # Hadamard suele ser más largo
        elif operacion.tipo == OperacionCuantica.FASE_S:
             params["virtual"] = True
             params["angulo_logico"] = math.pi / 2 # Rotación Z de 90 grados
             params["amplitud"] = 0.0
        
        params["amplitud"] *= self.calibracion["factor_amplitud"]
        
        #logger.debug(f"Parámetros de Pulso para {operacion.tipo.name}: {params}")
        return params

    def aplicar_pulso(self, qubit: QubitSuperconductor, params: Dict[str, Any]) -> str:
        """Simula la aplicación del pulso al qubit."""
        # Simular latencia
        time.sleep(self.latencia_aplicacion * 1e-9)
        
        # Simular decoherencia durante el pulso (simplificado)
        qubit.simular_decoherencia() # Decoherencia natural justo antes
        tiempo_pulso_us = params.get("duracion", 0) * 1e-3 # ns a µs
        qubit.estado_complejo.normalize() # Renormalizar por si decoherencia afectó norma
        
        # Aplicar operación lógica
        tipo_op = params.get("tipo_operacion", "")
        resultado_op = "Éxito"
        angulo_op = params.get("angulo_logico") # Ángulo deseado

        try:
            # Simular error de control (dependiente de amplitud, duración?)
            error_control_prob = 0.01 + (1.0 - params.get("amplitud", 1.0)) * 0.05
            if np.random.random() < error_control_prob:
                 angulo_real = angulo_op * np.random.normal(1.0, 0.1) if angulo_op is not None else None # Error en ángulo
                 logger.warning(f"Error de control simulado! Ángulo aplicado: {angulo_real:.4f} vs deseado {angulo_op:.4f}")
            else:
                 angulo_real = angulo_op

            if params.get("virtual", False):
                 # Rotaciones virtuales Z o S
                 if angulo_real is not None: qubit.aplicar_rotacion('Z', angulo_real)
                 logger.debug(f"Pulso virtual {tipo_op} aplicado.")
            elif tipo_op == "ROTACION_X":
                 if angulo_real is not None: qubit.aplicar_rotacion('X', angulo_real)
            elif tipo_op == "ROTACION_Y":
                 if angulo_real is not None: qubit.aplicar_rotacion('Y', angulo_real)
            elif tipo_op == "HADAMARD_PULSE":
                 # Simular Hadamard con posible error
                 if np.random.random() > 0.02: # 2% de fallo en Hadamard
                      qubit.aplicar_hadamard()
                 else:
                      logger.error("Fallo simulado en compuerta Hadamard!")
                      resultado_op = "Fallo Hadamard"
            elif tipo_op == "RESET":
                 qubit.reset()
            elif tipo_op == "MEDICION":
                 med = qubit.medir()
                 resultado_op = f"Medido |{med}⟩"
            else:
                 logger.warning(f"Tipo de pulso/operación no manejado: {tipo_op}")
                 resultado_op = "Operación no implementada"

            # Simular decoherencia post-pulso
            qubit.simular_decoherencia()

        except Exception as e:
             logger.error(f"Excepción al aplicar pulso {tipo_op}: {e}")
             resultado_op = "Error Excepción"

        #logger.info(f"Pulso {tipo_op} aplicado a {qubit.id}. Resultado: {resultado_op}")
        return resultado_op

# --- Clases de Componentes Ópticos (Simulados) ---

@dataclass
class EstadoFoton:
    """Estado simplificado de un fotón óptico para comunicación."""
    polarizacion: float # Ángulo de polarización en radianes [0, pi]
    fase: float        # Fase relativa en radianes [0, 2*pi]
    # frecuencia: float # Frecuencia en THz (asumida constante)
    # intensidad: float # Intensidad (asumida 1 si existe, 0 si no)
    valido: bool = True # Indica si el fotón representa un estado válido

    def __str__(self) -> str:
        if not self.valido: return "Fotón Inválido/Perdido"
        return f"Fotón[pol={math.degrees(self.polarizacion):.1f}°, fase={math.degrees(self.fase):.1f}°]"

class TransductorSQaOptico:
    """Transductor simulado Superconductor -> Óptico."""

    def __init__(self, eficiencia_base: float = 0.8):
        self.eficiencia_conversion = eficiencia_base # Probabilidad de éxito en transducción
        self.ruido_fase_polarizacion = 0.05 # Radianes de ruido añadido
        logger.info(f"Transductor SQ->Óptico inicializado. Eficiencia base: {self.eficiencia_conversion:.2f}")

    def leer_estado_sq(self, qubit: QubitSuperconductor) -> Optional[EstadoComplejo]:
        """Lee el estado del qubit (simulado, podría ser destructivo o QND)."""
        # Simular posible fallo en lectura basado en SNR (obtenido de métricas?)
        snr_simulado = 25.0 # Valor fijo para simplificar aquí
        prob_fallo_lectura = 0.01 + (1.0 - np.clip(snr_simulado / 30.0, 0, 1)) * 0.1
        if np.random.random() < prob_fallo_lectura:
             logger.warning(f"Fallo simulado en lectura de estado de {qubit.id}")
             return None # Falla la lectura
        
        # Devolver copia del estado para no modificar el original si la lectura no es QND
        return EstadoComplejo(qubit.estado_complejo.alpha, qubit.estado_complejo.beta)

    def mapear_estado_a_foton(self, estado_sq: EstadoComplejo) -> EstadoFoton:
        """Mapea el estado del qubit al estado de un fotón (heurístico)."""
        # Mapeo:
        # - Fase relativa del qubit -> Fase del fotón
        # - Probabilidad P(|1>) -> Polarización del fotón (linealmente entre 0 y pi/2)
        
        fase_relativa_sq = estado_sq.fase_relativa()
        prob_1 = estado_sq.probabilidad_1()
        
        # Mapeo de probabilidad a polarización (0 a 90 grados)
        polarizacion_foton = prob_1 * (math.pi / 2.0)
        
        # Mapeo de fase relativa a fase del fotón
        fase_foton = (fase_relativa_sq + math.pi) % (2 * math.pi) # Mapear a [0, 2pi]

        # Añadir ruido simulado
        polarizacion_foton += np.random.normal(0, self.ruido_fase_polarizacion)
        fase_foton += np.random.normal(0, self.ruido_fase_polarizacion)
        
        # Asegurar rangos válidos
        polarizacion_foton = np.clip(polarizacion_foton, 0, math.pi)
        fase_foton = fase_foton % (2 * math.pi)
        
        return EstadoFoton(polarizacion=polarizacion_foton, fase=fase_foton)

    def modular_foton(self, estado_foton_deseado: EstadoFoton) -> Optional[EstadoFoton]:
        """Simula la creación y modulación de un fotón."""
        # Simular fallo de transducción basado en eficiencia
        if np.random.random() > self.eficiencia_conversion:
             logger.warning("Fallo simulado en transducción SQ -> Óptico")
             return None # Fotón no se crea/modula correctamente
        
        # Devolver el estado deseado (asumiendo modulación perfecta si no falla)
        logger.debug(f"Fotón modulado con estado: {estado_foton_deseado}")
        return estado_foton_deseado

class OpticalChannel:
    """Canal de comunicación óptico simulado."""
    def __init__(self, longitud_km: float = 1.0, atenuacion_db_km: float = 0.2):
        self.longitud = longitud_km
        self.atenuacion_db_km = atenuacion_db_km
        self.prob_perdida = 1.0 - 10**(-(self.longitud * self.atenuacion_db_km) / 10.0)
        self.latencia_ns = longitud_km * 5000 # 5 us/km -> ns/km
        self.fotones_perdidos = 0
        logger.info(f"Canal Óptico inicializado: {longitud_km}km, Aten={atenuacion_db_km}dB/km -> Prob Pérdida={self.prob_perdida:.3f}, Latencia={self.latencia_ns:.0f}ns")

    def enviar_foton(self, foton: Optional[EstadoFoton]) -> Optional[EstadoFoton]:
        """Simula el envío de un fotón por el canal."""
        # Simular latencia
        time.sleep(self.latencia_ns * 1e-9)
        
        if foton is None: # Si la modulación falló
            self.fotones_perdidos += 1
            logger.warning("Intento de enviar fotón NULO (fallo modulación previa).")
            return None

        # Simular pérdida por atenuación
        if np.random.random() < self.prob_perdida:
             self.fotones_perdidos += 1
             logger.warning(f"Fotón perdido en el canal! (Total perdidos: {self.fotones_perdidos})")
             return None # Fotón perdido
        else:
             # Simular ruido adicional en fase/polarización por el canal
             ruido_canal = 0.02 # Radianes
             foton.polarizacion = np.clip(foton.polarizacion + np.random.normal(0, ruido_canal), 0, math.pi)
             foton.fase = (foton.fase + np.random.normal(0, ruido_canal)) % (2 * math.pi)
             #logger.debug(f"Fotón transmitido exitosamente por el canal.")
             return foton

    @property
    def perdida_acumulada(self) -> float:
        """Ratio de fotones perdidos hasta ahora."""
        total_intentos = self.fotones_perdidos + (ciclo_actual if 'ciclo_actual' in globals() else 0) # Necesita contador global
        return self.fotones_perdidos / total_intentos if total_intentos > 0 else 0.0


class PhotonDetector:
    """Detector de fotones simulado."""
    def __init__(self, eficiencia_deteccion: float = 0.9, dark_counts_hz: float = 100.0, error_medicion_rad: float = 0.1):
        self.eficiencia = eficiencia_deteccion
        self.dark_count_rate_s = dark_counts_hz
        self.prob_dark_count_per_ns = dark_counts_hz * 1e-9
        self.error_medicion = error_medicion_rad # Error en radianes para fase/polarización
        self.fotones_detectados = 0
        self.dark_counts_registrados = 0
        logger.info(f"Detector Fotones inicializado. Eficiencia={self.eficiencia:.2f}, DarkCounts={dark_counts_hz}Hz, ErrorMed={math.degrees(self.error_medicion):.1f}°")

    def detectar(self, foton_entrante: Optional[EstadoFoton], ventana_tiempo_ns: float = 10.0) -> Tuple[Optional[EstadoFoton], bool, bool]:
        """Simula la detección de un fotón."""
        # Simular dark counts
        prob_dark = ventana_tiempo_ns * self.prob_dark_count_per_ns
        es_dark_count = np.random.random() < prob_dark
        if es_dark_count:
            self.dark_counts_registrados += 1
            logger.warning(f"Dark Count detectado! (Total: {self.dark_counts_registrados})")
            # Devolver un estado "falso" o None? Devolver None y flag.
            return None, False, True # (No hay fotón real, No éxito detección, Sí es dark count)

        # Si no hubo dark count, ver si llegó un fotón real
        if foton_entrante is None:
             return None, False, False # (No llegó fotón, No éxito, No dark count)

        # Simular eficiencia de detección
        if np.random.random() < self.eficiencia:
             # Fotón detectado! Simular error de medición
             self.fotones_detectados += 1
             polarizacion_medida = foton_entrante.polarizacion + np.random.normal(0, self.error_medicion)
             fase_medida = foton_entrante.fase + np.random.normal(0, self.error_medicion)
             
             foton_medido = EstadoFoton(
                 polarizacion=np.clip(polarizacion_medida, 0, math.pi),
                 fase=fase_medida % (2 * math.pi),
                 valido=True
             )
             #logger.debug(f"Fotón detectado y medido: {foton_medido}")
             return foton_medido, True, False # (Fotón medido, Éxito detección, No dark count)
        else:
             #logger.warning("Fotón llegó pero no fue detectado (eficiencia).")
             return None, False, False # (Fotón no detectado, No éxito, No dark count)


class TransductorOpticoAClasico:
    """Convierte estado de fotón medido a información clásica."""
    def __init__(self, umbral_polarizacion: float = math.pi / 4.0): # Umbral a 45 grados
        self.umbral = umbral_polarizacion
        logger.info(f"Transductor Óptico->Clásico inicializado. Umbral Pol={math.degrees(self.umbral):.1f}°")

    def decodificar_estado(self, foton_medido: Optional[EstadoFoton]) -> Optional[int]:
        """Decodifica el estado del fotón a un bit clásico (0 o 1)."""
        if foton_medido is None or not foton_medido.valido:
            return None # No se puede decodificar

        # Decodificar basado en polarización (ejemplo simple)
        # Si polarización < umbral -> Bit 0
        # Si polarización >= umbral -> Bit 1
        bit_decodificado = 1 if foton_medido.polarizacion >= self.umbral else 0
        
        #logger.debug(f"Fotón {foton_medido} decodificado a Bit: {bit_decodificado}")
        return bit_decodificado

# --- Ciclo Principal de Simulación ---

def ejecutar_simulacion(num_ciclos: int = 100, intervalo_s: float = 0.1):
    """Ejecuta el ciclo principal de la simulación."""
    
    logger.info("--- Iniciando Simulación QuoreMind Híbrida ---")
    
    # Inicializar componentes
    qubit = QubitSuperconductor(id_qubit="QSim")
    control_mente = QuoreMind()
    control_microondas = MicrowaveControl()
    transductor_sq_opt = TransductorSQaOptico(eficiencia_base=0.8)
    canal_optico = OpticalChannel(longitud_km=0.5, atenuacion_db_km=0.25)
    detector_fotones = PhotonDetector(eficiencia_deteccion=0.9, dark_counts_hz=50, error_medicion_rad=0.08)
    decodificador_opt_clasico = TransductorOpticoAClasico(umbral_polarizacion=math.pi/4)
    
    global ciclo_actual # Para referencia en OpticalChannel.perdida_acumulada
    
    for ciclo in range(1, num_ciclos + 1):
        ciclo_actual = ciclo
        logger.info(f"\n--- Iniciando Ciclo {ciclo}/{num_ciclos} ---")
        
        # 1. Obtener Métricas
        metricas = control_mente.obtener_metricas_actuales(qubit, canal_optico, detector_fotones)
        
        # 2. Decidir Operación
        operacion = control_mente.decidir_operacion(metricas)
        
        # 3. Traducir y Aplicar Pulso
        params_pulso = control_microondas.traducir_operacion_a_pulso(operacion)
        resultado_pulso = control_microondas.aplicar_pulso(qubit, params_pulso)
        
        # 4. Leer Estado SQ (si la operación no fue medición/reset)
        estado_sq_leido = None
        if operacion.tipo not in [OperacionCuantica.MEDICION, OperacionCuantica.RESET]:
            estado_sq_leido = transductor_sq_opt.leer_estado_sq(qubit)
        
        # 5. Mapear a Fotón y Modular
        foton_modulado = None
        if estado_sq_leido:
            estado_foton_deseado = transductor_sq_opt.mapear_estado_a_foton(estado_sq_leido)
            foton_modulado = transductor_sq_opt.modular_foton(estado_foton_deseado)
            if not foton_modulado:
                 control_mente.registrar_error("Fallo en modulación/transducción")
        
        # 6. Enviar por Canal Óptico
        foton_recibido_canal = canal_optico.enviar_foton(foton_modulado)
        
        # 7. Detectar Fotón
        # Simular ventana de tiempo para detección
        ventana_deteccion_ns = canal_optico.latencia_ns + 5.0 # Latencia + margen
        foton_detectado_estado, exito_det, fue_dark_count = detector_fotones.detectar(
            foton_recibido_canal, ventana_deteccion_ns
        )
        
        # 8. Decodificar
        bit_recibido = decodificador_opt_clasico.decodificar_estado(foton_detectado_estado)
        
        # 9. Registrar Resultado y Actualizar Aprendizaje
        resultado_ciclo = {
            "ciclo": ciclo,
            "operacion_decidida": operacion.tipo.name,
            "angulo_calculado": operacion.angulo,
            "resultado_pulso": resultado_pulso,
            "estado_qubit_final": str(qubit),
            "foton_modulado": str(foton_modulado) if foton_modulado else "Fallo Modulación",
            "foton_recibido_canal": bool(foton_recibido_canal),
            "exito_deteccion": exito_det,
            "dark_count": fue_dark_count,
            "foton_medido": str(foton_detectado_estado) if foton_detectado_estado else "No Detectado",
            "bit_recibido": bit_recibido if bit_recibido is not None else "N/A"
        }
        control_mente.actualizar_aprendizaje(resultado_ciclo)
        
        # Loguear resumen del ciclo
        log_resumen = (f"Ciclo {ciclo}: Op={operacion.tipo.name}, "
                       f"Qubit={qubit.estado_complejo.alpha:.2f},{qubit.estado_complejo.beta:.2f}, "
                       f"Fotón Recibido={exito_det}, Bit={bit_recibido}")
        logger.info(log_resumen)
        
        # Opcional: Resetear qubit periódicamente o si decoherencia es alta
        if ciclo % 15 == 0: # Reset cada 15 ciclos
             logger.info(f"Reseteo periódico del qubit en ciclo {ciclo}")
             control_microondas.aplicar_pulso(qubit, {"tipo_operacion": "RESET"})

        # Esperar antes del siguiente ciclo
        time.sleep(intervalo_s)

    logger.info("--- Simulación Finalizada ---")
    estado_final_sistema = control_mente.obtener_estado_sistema()
    logger.info(f"Estado Final del Sistema: {estado_final_sistema}")
    logger.info(f"Total Fotones Perdidos en Canal: {canal_optico.fotones_perdidos}")
    logger.info(f"Total Dark Counts Detectados: {detector_fotones.dark_counts_registrados}")
    logger.info(f"Total Fotones Detectados Exitosamente: {detector_fotones.fotones_detectados}")

# --- Punto de Entrada ---
if __name__ == "__main__":
    # Definir variable global para referencia en OpticalChannel
    ciclo_actual = 0
    ejecutar_simulacion(num_ciclos=50, intervalo_s=0.05) # Ejecutar 50 ciclos con pausa corta

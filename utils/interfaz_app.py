import sys
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QLineEdit,
                            QComboBox, QTabWidget, QCheckBox, QSpinBox, 
                            QDoubleSpinBox, QTableWidget, QTableWidgetItem,
                            QFileDialog, QGroupBox, QGridLayout, QMessageBox,
                            QSplitter, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer

# Importamos nuestro integrador
from integration_with_interface import AgentInterfaceIntegrator

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MatplotlibCanvas(FigureCanvas):
    """Canvas de Matplotlib para gráficos en Qt"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        self.fig.tight_layout()


class QuantumAgentInterface(QMainWindow):
    """
    Interfaz gráfica principal para interactuar con el AgentInterfaceIntegrator
    """
    
    def __init__(self):
        super().__init__()
        
        # Inicializa el integrador de backend-frontend
        self.integrator = AgentInterfaceIntegrator()
        
        # Configuración inicial
        self.setWindowTitle("Quantum Agent Interface")
        self.setMinimumSize(1200, 800)
        
        # Crear y configurar la interfaz
        self._create_ui()
        
        # Configuración por defecto
        self._load_default_settings()
        
        # Timer para actualizar mediciones
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_quantum_displays)
        
        logger.info("Interfaz de usuario inicializada")
    
    def _create_ui(self):
        """Crea la interfaz de usuario completa"""
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(main_widget)
        
        # Tabs para organizar la interfaz
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Crear pestañas
        self._create_configuration_tab()
        self._create_quantum_circuit_tab()
        self._create_simulation_tab()
        self._create_analysis_tab()
        
        # Área de log en la parte inferior
        self._create_log_area(main_layout)
        
        # Botones de acción generales
        self._create_action_buttons(main_layout)
    
    def _create_configuration_tab(self):
        """Crea la pestaña de configuración del agente"""
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # Campos de configuración del agente
        agent_group = QGroupBox("Configuración del Agente Cuántico")
        agent_layout = QGridLayout(agent_group)
        
        # Nombre del agente
        agent_layout.addWidget(QLabel("Nombre:"), 0, 0)
        self.agent_name_input = QLineEdit("QuantumAgent")
        agent_layout.addWidget(self.agent_name_input, 0, 1)
        
        # Número de qubits
        agent_layout.addWidget(QLabel("Número de Qubits:"), 1, 0)
        self.num_qubits_spin = QSpinBox()
        self.num_qubits_spin.setRange(1, 10)
        self.num_qubits_spin.setValue(4)
        agent_layout.addWidget(self.num_qubits_spin, 1, 1)
        
        # Tasa de aprendizaje
        agent_layout.addWidget(QLabel("Tasa de Aprendizaje:"), 2, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.001, 1.0)
        self.learning_rate_spin.setSingleStep(0.01)
        self.learning_rate_spin.setValue(0.1)
        agent_layout.addWidget(self.learning_rate_spin, 2, 1)
        
        # Tamaño de entrada
        agent_layout.addWidget(QLabel("Tamaño de Entrada:"), 3, 0)
        self.input_size_spin = QSpinBox()
        self.input_size_spin.setRange(1, 100)
        self.input_size_spin.setValue(5)
        agent_layout.addWidget(self.input_size_spin, 3, 1)
        
        # Tamaño oculto
        agent_layout.addWidget(QLabel("Tamaño Oculto:"), 4, 0)
        self.hidden_size_spin = QSpinBox()
        self.hidden_size_spin.setRange(8, 256)
        self.hidden_size_spin.setValue(64)
        agent_layout.addWidget(self.hidden_size_spin, 4, 1)
        
        # Tipo de RNN
        agent_layout.addWidget(QLabel("Tipo de RNN:"), 5, 0)
        self.rnn_type_combo = QComboBox()
        self.rnn_type_combo.addItems(["GRU", "LSTM", "RNN"])
        agent_layout.addWidget(self.rnn_type_combo, 5, 1)
        
        # Tasa de dropout
        agent_layout.addWidget(QLabel("Tasa de Dropout:"), 6, 0)
        self.dropout_rate_spin = QDoubleSpinBox()
        self.dropout_rate_spin.setRange(0.0, 0.9)
        self.dropout_rate_spin.setSingleStep(0.05)
        self.dropout_rate_spin.setValue(0.2)
        agent_layout.addWidget(self.dropout_rate_spin, 6, 1)
        
        # Batch normalization
        agent_layout.addWidget(QLabel("Usar Batch Normalization:"), 7, 0)
        self.batch_norm_check = QCheckBox()
        self.batch_norm_check.setChecked(True)
        agent_layout.addWidget(self.batch_norm_check, 7, 1)
        
        # Decaimiento adaptativo
        agent_layout.addWidget(QLabel("Decaimiento Adaptativo LR:"), 8, 0)
        self.lr_decay_spin = QDoubleSpinBox()
        self.lr_decay_spin.setRange(0.8, 0.999)
        self.lr_decay_spin.setSingleStep(0.01)
        self.lr_decay_spin.setValue(0.95)
        agent_layout.addWidget(self.lr_decay_spin, 8, 1)
        
        config_layout.addWidget(agent_group)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        
        self.init_agent_btn = QPushButton("Inicializar Agente")
        self.init_agent_btn.clicked.connect(self.initialize_agent)
        buttons_layout.addWidget(self.init_agent_btn)
        
        self.load_config_btn = QPushButton("Cargar Configuración")
        self.load_config_btn.clicked.connect(self.load_configuration)
        buttons_layout.addWidget(self.load_config_btn)
        
        self.save_config_btn = QPushButton("Guardar Configuración")
        self.save_config_btn.clicked.connect(self.save_configuration)
        buttons_layout.addWidget(self.save_config_btn)
        
        config_layout.addLayout(buttons_layout)
        
        # Agregar pestaña al tabwidget
        self.tabs.addTab(config_tab, "Configuración")
    
    def _create_quantum_circuit_tab(self):
        """Crea la pestaña de circuito cuántico"""
        quantum_tab = QWidget()
        quantum_layout = QVBoxLayout(quantum_tab)
        
        # Sección superior: controles del circuito
        controls_group = QGroupBox("Control del Circuito Cuántico")
        controls_layout = QGridLayout(controls_group)
        
        # Número de qubits
        controls_layout.addWidget(QLabel("Número de Qubits:"), 0, 0)
        self.circuit_qubits_spin = QSpinBox()
        self.circuit_qubits_spin.setRange(1, 10)
        self.circuit_qubits_spin.setValue(4)
        controls_layout.addWidget(self.circuit_qubits_spin, 0, 1)
        
        # Inicializar circuito
        self.init_circuit_btn = QPushButton("Inicializar Circuito")
        self.init_circuit_btn.clicked.connect(self.initialize_quantum_circuit)
        controls_layout.addWidget(self.init_circuit_btn, 0, 2)
        
        # Gates comunes
        controls_layout.addWidget(QLabel("Aplicar Compuerta:"), 1, 0)
        self.gate_combo = QComboBox()
        self.gate_combo.addItems(["H (Hadamard)", "X (NOT)", "Y", "Z", "S", "T", 
                                 "CNOT", "SWAP", "RX", "RY", "RZ"])
        controls_layout.addWidget(self.gate_combo, 1, 1)
        
        # Qubit(s) para la compuerta
        controls_layout.addWidget(QLabel("Qubit(s):"), 2, 0)
        self.qubit_input = QLineEdit("0")
        controls_layout.addWidget(self.qubit_input, 2, 1)
        controls_layout.addWidget(QLabel("(Separar múltiples qubits con comas)"), 2, 2)
        
        # Parámetro para compuertas parametrizadas
        controls_layout.addWidget(QLabel("Parámetro:"), 3, 0)
        self.gate_param_spin = QDoubleSpinBox()
        self.gate_param_spin.setRange(-6.28, 6.28)
        self.gate_param_spin.setSingleStep(0.1)
        self.gate_param_spin.setValue(0.0)
        controls_layout.addWidget(self.gate_param_spin, 3, 1)
        
        # Botón para aplicar compuerta
        self.apply_gate_btn = QPushButton("Aplicar Compuerta")
        self.apply_gate_btn.clicked.connect(self.apply_quantum_gate)
        controls_layout.addWidget(self.apply_gate_btn, 4, 1)
        
        # Botones para medición
        measure_layout = QHBoxLayout()
        self.measure_qubit_btn = QPushButton("Medir Qubit")
        self.measure_qubit_btn.clicked.connect(self.measure_specific_qubit)
        measure_layout.addWidget(self.measure_qubit_btn)
        
        self.measure_all_btn = QPushButton("Medir Todos")
        self.measure_all_btn.clicked.connect(self.measure_all_qubits)
        measure_layout.addWidget(self.measure_all_btn)
        
        controls_layout.addLayout(measure_layout, 5, 1)
        
        quantum_layout.addWidget(controls_group)
        
        # Sección inferior: visualización del estado cuántico
        display_group = QGroupBox("Estado Cuántico")
        display_layout = QHBoxLayout(display_group)
        
        # Splitter para dividir visualizaciones
        splitter = QSplitter(Qt.Horizontal)
        
        # Gráfico de amplitudes
        amplitude_container = QWidget()
        amplitude_layout = QVBoxLayout(amplitude_container)
        amplitude_layout.addWidget(QLabel("Amplitudes"))
        
        self.amplitude_canvas = MatplotlibCanvas(amplitude_container, width=5, height=4)
        amplitude_layout.addWidget(self.amplitude_canvas)
        
        # Gráfico de probabilidades
        probability_container = QWidget()
        probability_layout = QVBoxLayout(probability_container)
        probability_layout.addWidget(QLabel("Probabilidades"))
        
        self.probability_canvas = MatplotlibCanvas(probability_container, width=5, height=4)
        probability_layout.addWidget(self.probability_canvas)
        
        splitter.addWidget(amplitude_container)
        splitter.addWidget(probability_container)
        display_layout.addWidget(splitter)
        
        quantum_layout.addWidget(display_group)
        
        # Agregar pestaña al tabwidget
        self.tabs.addTab(quantum_tab, "Circuito Cuántico")
    
    def _create_simulation_tab(self):
        """Crea la pestaña de simulación"""
        sim_tab = QWidget()
        sim_layout = QVBoxLayout(sim_tab)
        
        # Controles de simulación
        sim_controls = QGroupBox("Parámetros de Simulación")
        controls_layout = QGridLayout(sim_controls)
        
        # Carga de datos
        controls_layout.addWidget(QLabel("Datos de Entrada:"), 0, 0)
        self.data_file_input = QLineEdit()
        controls_layout.addWidget(self.data_file_input, 0, 1)
        
        self.load_data_btn = QPushButton("Cargar Datos")
        self.load_data_btn.clicked.connect(self.load_simulation_data)
        controls_layout.addWidget(self.load_data_btn, 0, 2)
        
        # Longitud de secuencia
        controls_layout.addWidget(QLabel("Longitud de Secuencia:"), 1, 0)
        self.seq_length_spin = QSpinBox()
        self.seq_length_spin.setRange(1, 100)
        self.seq_length_spin.setValue(10)
        controls_layout.addWidget(self.seq_length_spin, 1, 1)

        # Número de iteraciones
        controls_layout.addWidget(QLabel("Número de Iteraciones:"), 2, 0)
        self.num_iterations_spin = QSpinBox()
        self.num_iterations_spin.setRange(1, 1000)
        self.num_iterations_spin.setValue(50)
        controls_layout.addWidget(self.num_iterations_spin, 2, 1)
        
        # Épocas de entrenamiento
        controls_layout.addWidget(QLabel("Épocas de Entrenamiento:"), 3, 0)
        self.training_epochs_spin = QSpinBox()
        self.training_epochs_spin.setRange(1, 500)
        self.training_epochs_spin.setValue(20)
        controls_layout.addWidget(self.training_epochs_spin, 3, 1)

        # Tamaño de lote
        controls_layout.addWidget(QLabel("Tamaño de Lote:"), 4, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(32)
        controls_layout.addWidget(self.batch_size_spin, 4, 1)

        # Botón de ejecución
        self.run_simulation_btn = QPushButton("Ejecutar Simulación")
        self.run_simulation_btn.clicked.connect(self.run_simulation)
        controls_layout.addWidget(self.run_simulation_btn, 5, 1)

        sim_layout.addWidget(sim_controls)

        # Resultados de la simulación (temporal)
        results_group = QGroupBox("Resultados de la Simulación")
        results_layout = QVBoxLayout(results_group)
        self.results_label = QLabel("Resultados aquí...")
        results_layout.addWidget(self.results_label)
        sim_layout.addWidget(results_group)

        # Agregar pestaña al tabwidget
        self.tabs.addTab(sim_tab, "Simulación")

    def _create_analysis_tab(self):
        """Crea la pestaña de análisis estadístico y bayesiano."""
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)

        # Grupo para análisis estadístico
        stats_group = QGroupBox("Análisis Estadístico")
        stats_layout = QGridLayout(stats_group)

        # Datos para análisis estadístico
        stats_layout.addWidget(QLabel("Datos (CSV):"), 0, 0)
        self.stats_data_input = QLineEdit()
        stats_layout.addWidget(self.stats_data_input, 0, 1)
        self.load_stats_data_btn = QPushButton("Cargar Datos")
        self.load_stats_data_btn.clicked.connect(self.load_statistical_data)
        stats_layout.addWidget(self.load_stats_data_btn, 0, 2)

        # Punto para Mahalanobis
        stats_layout.addWidget(QLabel("Punto para Mahalanobis:"), 1, 0)
        self.mahalanobis_point_input = QLineEdit()
        stats_layout.addWidget(self.mahalanobis_point_input, 1, 1)
        
        # Botón para realizar análisis
        self.analyze_stats_btn = QPushButton("Analizar")
        self.analyze_stats_btn.clicked.connect(self.perform_statistical_analysis)
        stats_layout.addWidget(self.analyze_stats_btn, 2, 1)

        # Resultados del análisis estadístico
        stats_layout.addWidget(QLabel("Matriz de Covarianza:"), 3, 0)
        self.covariance_table = QTableWidget()
        self.covariance_table.setEditTriggers(QTableWidget.NoEditTriggers) #Tabla solo de lectura
        stats_layout.addWidget(self.covariance_table, 3, 1)

        stats_layout.addWidget(QLabel("Distancia de Mahalanobis:"), 4, 0)
        self.mahalanobis_result_label = QLabel("")
        stats_layout.addWidget(self.mahalanobis_result_label, 4, 1)

        analysis_layout.addWidget(stats_group)

        # Grupo para análisis bayesiano
        bayes_group = QGroupBox("Análisis Bayesiano")
        bayes_layout = QGridLayout(bayes_group)

        # Entropía
        bayes_layout.addWidget(QLabel("Entropía:"), 0, 0)
        self.entropy_input = QDoubleSpinBox()
        self.entropy_input.setRange(0.0, 5.0)
        self.entropy_input.setSingleStep(0.1)
        bayes_layout.addWidget(self.entropy_input, 0, 1)

        # Coherencia
        bayes_layout.addWidget(QLabel("Coherencia:"), 1, 0)
        self.coherence_input = QDoubleSpinBox()
        self.coherence_input.setRange(0.0, 1.0)
        self.coherence_input.setSingleStep(0.1)
        bayes_layout.addWidget(self.coherence_input, 1, 1)

        # Influencia PRN
        bayes_layout.addWidget(QLabel("Influencia PRN:"), 2, 0)
        self.prn_influence_input = QDoubleSpinBox()
        self.prn_influence_input.setRange(0.0, 1.0)
        self.prn_influence_input.setSingleStep(0.1)
        bayes_layout.addWidget(self.prn_influence_input, 2, 1)

        # Acción
        bayes_layout.addWidget(QLabel("Acción:"), 3, 0)
        self.bayes_action_combo = QComboBox()
        self.bayes_action_combo.addItems(["0", "1"])
        bayes_layout.addWidget(self.bayes_action_combo, 3, 1)

         # Botón para calcular
        self.calculate_bayes_btn = QPushButton("Calcular")
        self.calculate_bayes_btn.clicked.connect(self.calculate_bayesian_metrics)
        bayes_layout.addWidget(self.calculate_bayes_btn, 4, 1)

        # Resultados del análisis bayesiano
        bayes_layout.addWidget(QLabel("Resultados:"), 5, 0)
        self.bayes_results_table = QTableWidget()
        self.bayes_results_table.setEditTriggers(QTableWidget.NoEditTriggers) # Tabla solo de lectura
        self.bayes_results_table.setColumnCount(2)
        self.bayes_results_table.setHorizontalHeaderLabels(["Métrica", "Valor"])
        bayes_layout.addWidget(self.bayes_results_table, 5, 1)


        analysis_layout.addWidget(bayes_group)
        self.tabs.addTab(analysis_tab, "Análisis")

    def _create_log_area(self, parent_layout):
        """Crea el área de registro (log) de la interfaz"""
        log_group = QGroupBox("Log de Eventos")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        parent_layout.addWidget(log_group)

    def _create_action_buttons(self, parent_layout):
        """
        Crea botones de acción globales como 'Cargar/Guardar Agente'.
        """
        buttons_layout = QHBoxLayout()

        self.load_agent_btn = QPushButton("Cargar Agente")
        self.load_agent_btn.clicked.connect(self.load_agent)
        buttons_layout.addWidget(self.load_agent_btn)

        self.save_agent_btn = QPushButton("Guardar Agente")
        self.save_agent_btn.clicked.connect(self.save_agent)
        buttons_layout.addWidget(self.save_agent_btn)

        parent_layout.addLayout(buttons_layout)

    def _load_default_settings(self):
        """Carga la configuración por defecto en la interfaz"""
        # Aquí puedes definir valores por defecto más específicos si lo deseas
        pass
    
    @pyqtSlot()
    def initialize_agent(self):
        """Inicializa el agente con la configuración de la interfaz"""
        try:
            agent_settings = {
                'name': self.agent_name_input.text(),
                'num_qubits': self.num_qubits_spin.value(),
                'learning_rate': self.learning_rate_spin.value(),
                'input_size': self.input_size_spin.value(),
                'hidden_size': self.hidden_size_spin.value(),
                'rnn_type': self.rnn_type_combo.currentText(),
                'dropout_rate': self.dropout_rate_spin.value(),
                'use_batch_norm': self.batch_norm_check.isChecked(),
                'adaptative_lr_decay': self.lr_decay_spin.value()
            }
            self.integrator.setup_agent(agent_settings)
            self.log_message("Agente cuántico inicializado con éxito.")
        except Exception as e:
            self.show_error_message(f"Error al inicializar el agente: {e}")
            logger.exception("Error al inicializar el agente")

    @pyqtSlot()
    def load_configuration(self):
        """Carga una configuración desde un archivo"""
        filepath, _ = QFileDialog.getOpenFileName(self, "Cargar Configuración", "", "JSON Files (*.json)")
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    config = json.load(f)
                # Aquí debes validar y aplicar la configuración cargada
                # a los widgets correspondientes de la interfaz. Por ejemplo:

                # Configuracion del agente
                self.agent_name_input.setText(config.get('agent_name', 'QuantumAgent'))
                self.num_qubits_spin.setValue(config.get('num_qubits', 4))
                self.learning_rate_spin.setValue(config.get('learning_rate', 0.1))
                self.input_size_spin.setValue(config.get('input_size', 5))
                self.hidden_size_spin.setValue(config.get('hidden_size', 64))
                self.rnn_type_combo.setCurrentText(config.get('rnn_type', 'GRU'))
                self.dropout_rate_spin.setValue(config.get('dropout_rate', 0.2))
                self.batch_norm_check.setChecked(config.get('use_batch_norm', True))
                self.lr_decay_spin.setValue(config.get('adaptative_lr_decay', 0.99))

                #Configuración del circuito
                self.circuit_qubits_spin.setValue(config.get('circuit_qubits', 4))

                #Configuración de la simulación
                self.data_file_input.setText(config.get('data_path', ''))
                self.seq_length_spin.setValue(config.get('sequence_length', 10))
                self.num_iterations_spin.setValue(config.get('num_iterations', 50))
                self.training_epochs_spin.setValue(config.get('training_epochs', 20))
                self.batch_size_spin.setValue(config.get('batch_size', 32))

                self.log_message(f"Configuración cargada desde {filepath}")
            except Exception as e:
                self.show_error_message(f"Error al cargar la configuración: {e}")
                logger.exception("Error al cargar la configuracion")
    
    @pyqtSlot()
    def save_configuration(self):
        """Guarda la configuración actual en un archivo"""
        filepath, _ = QFileDialog.getSaveFileName(self, "Guardar Configuración", "", "JSON Files (*.json)")
        if filepath:
            try:
                config = {
                    'agent_name': self.agent_name_input.text(),
                    'num_qubits': self.num_qubits_spin.value(),
                    'learning_rate': self.learning_rate_spin.value(),
                    'input_size': self.input_size_spin.value(),
                    'hidden_size': self.hidden_size_spin.value(),
                    'rnn_type': self.rnn_type_combo.currentText(),
                    'dropout_rate': self.dropout_rate_spin.value(),
                    'use_batch_norm': self.batch_norm_check.isChecked(),
                    'adaptative_lr_decay': self.lr_decay_spin.value(),
                    'circuit_qubits': self.circuit_qubits_spin.value(),
                    'data_path': self.data_file_input.text(),
                    'sequence_length': self.seq_length_spin.value(),
                    'num_iterations': self.num_iterations_spin.value(),
                    'training_epochs': self.training_epochs_spin.value(),
                    'batch_size': self.batch_size_spin.value()
                }
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=4)
                self.log_message(f"Configuración guardada en {filepath}")
            
            except Exception as e:
                self.show_error_message(f"Error al guardar la configuración: {e}")
                logger.exception("Error al guardar la configuracion")

    @pyqtSlot()
    def initialize_quantum_circuit(self):
        """Inicializa el circuito cuántico con el número de qubits especificado"""
        num_qubits = self.circuit_qubits_spin.value()
        try:
            self.integrator.setup_quantum_circuit(num_qubits)
            self.log_message(f"Circuito cuántico inicializado con {num_qubits} qubits.")
            self.update_quantum_displays()  # Actualiza las visualizaciones
            self.update_timer.start(1000)  # Actualiza cada segundo
        except Exception as e:
            self.show_error_message(f"Error al inicializar el circuito: {e}")
            logger.exception("Error al inicializar el circuito cuantico")

    @pyqtSlot()
    def apply_quantum_gate(self):
        """Aplica una compuerta cuántica al circuito"""
        gate_text = self.gate_combo.currentText()
        qubit_str = self.qubit_input.text()
        param = self.gate_param_spin.value()
        
        try:
            qubits = [int(q.strip()) for q in qubit_str.split(",")]
            gate_name = gate_text.split(" ")[0]  # Obtiene el nombre de la compuerta
            
            # Construye la lista de compuertas para apply_custom_gates
            if gate_name in ["RX", "RY", "RZ"]:
                gates_list = [(gate_name.lower(), [param], qubits)]
            else:
                gates_list = [(gate_name.lower(), None, qubits)]
            
            self.integrator.apply_custom_gates(gates_list)
            self.log_message(f"Compuerta {gate_text} aplicada a qubit(s) {qubit_str}")
            self.update_quantum_displays()  # Actualiza visualizaciones
        
        except Exception as e:
            self.show_error_message(f"Error al aplicar la compuerta: {e}")
            logger.exception("Error al aplicar compuerta cuantica")

    @pyqtSlot()
    def measure_specific_qubit(self):
        """Mide un qubit específico del circuito"""
        try:
            qubit_str = self.qubit_input.text()
            qubit_index = int(qubit_str.strip())  # Asume que se introduce un solo qubit
            self.integrator.measure_qubit(qubit_index)
            self.log_message(f"Qubit {qubit_index} medido.")
            self.update_quantum_displays()  # Actualiza visualizaciones

        except Exception as e:
            self.show_error_message(f"Error al medir el qubit: {e}")
            logger.exception("Error al medir el Qubit")

    @pyqtSlot()
    def measure_all_qubits(self):
        """Mide todos los qubits del circuito"""
        try:
            self.integrator.measure_all_qubits()
            self.log_message("Todos los qubits medidos.")
            self.update_quantum_displays()  # Actualiza las visualizaciones

        except Exception as e:
            self.show_error_message("Error al medir todos los qubits.")
            logger.exception("Error al medir todos los qubits")
    
    def update_quantum_displays(self):
        """Actualiza los gráficos de amplitudes y probabilidades"""
        try:
            if self.integrator.resilient_circuit is not None:
                # Obtener

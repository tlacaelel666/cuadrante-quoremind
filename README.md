# cuadrante
redes y circuito cuiantico
# Documentacion
   # Clase ObjetoBinario: Se mantiene sin cambios, ya que seguirá siendo parte del entorno simulado.

   # Clase QNetwork:

        Define la red neuronal que se usará como función Q para el agente RL. La red mapea estados a valores de Q.

   # Clase EntornoSimulado:
        __init__: Recibe la lista de objetos binarios y define el estado inicial.

        obtener_estado: Devuelve el índice del objeto actual como el estado del entorno.

   # ejecutar_accion: Realiza una acción y devuelve el nuevo estado, la recompensa y el estado actual. Aquí están definidas las acciones:
            0: Mover a la derecha.

            1: Mover a la izquierda.

            2: Incrementar la subcategoría 1.

            3: Decrementar la subcategoría 1.

        obtener_texto_estado: Devuelve una descripción del estado actual.

  #  Clase Aplicacion:

        __init__: Inicializa el entorno simulado, la red neuronal Q, el optimizador, y define los parámetros para el agente de RL.

        crear_interfaz: Crea los widgets de Tkinter:

            ttk.Entry para la entrada de texto.

            ttk.Button para enviar el comando.

            scrolledtext.ScrolledText para mostrar la retroalimentación.

  #      procesar_comando:

            Obtiene el texto del usuario.

            Llama a interpretar_comando para obtener la acción correspondiente.

            Ejecuta la acción en el entorno simulado usando el método ejecutar_accion.

            Muestra la retroalimentación en el scrolledtext.

            Llama a la función aprender para entrenar el agente RL.

   #     interpretar_comando: Usa expresiones regulares para interpretar la intención del comando del usuario de forma sencilla. Si no reconoce el comando, se elige una acción aleatoria.

   #     aprender: Realiza el aprendizaje mediante Q-Learning, actualizando la red Q con la diferencia entre el valor Q esperado y el valor actual.

   #    seleccionar_accion: Implementa una política epsilon-greedy, seleccionando una acción aleatoria con probabilidad epsilon, o usando la mejor acción estimada por la red Q.

   #    entrenar_agente: Realiza una cantidad de interacciones con el entorno para que la red Q pueda aprender y mejore la toma de decisiones.

# Puntos Clave:

    PLN Simplificado: Se usa re.search para entender los comandos. En lugar de una librería más avanzada, se optó por un sistema de reglas básicas.

  #  Entorno Simulado Simple: El entorno se reduce al objeto actual y la capacidad de modificar su subcategoría 1 y moverse entre objetos.

  #  Aprendizaje por Refuerzo: Se usa el algoritmo Q-Learning como un enfoque simple pero eficaz. El agente aprende con la retroalimentación de las recompensas.

    Interacción: La interacción es basada en texto, con retroalimentación textual del entorno simulado.

# Cómo Usar:

   # Ejecuta el código.

    Escribe comandos en la entrada de texto, como: "izquierda", "derecha", "aumenta", "disminuye".

    Haz clic en "Enviar" para que el comando se procese y el agente RL tome la acción.

    La retroalimentación se mostrará en el área de texto, mostrando la acción realizada, el estado actual y la recompensa obtenida.

    Haz clic en el boton "Entrenar" para que la red neuronal que utiliza el agente RL aprenda.

   # Clase ActorCritic:

        Combina el actor (política) y el critic (valor) en una sola clase.

        Utiliza capas totalmente conectadas para la arquitectura de las redes.

  # Integración de A2C:

        almacenar_experiencia: Almacena los logaritmos de las probabilidades de las acciones, los valores estimados por el critic y las recompensas.

        calcular_retorno: Calcula el retorno acumulado (recompensas descontadas) para el entrenamiento.

        actualizar_red: Actualiza la red del actor y el critic usando las ventajas y los retornos acumulados.

        seleccionar_accion: Ahora usa el actor para seleccionar acciones probabilísticamente.

        entrenar_agente: Se ha modificado para que use la funcion almacenar_experiencia durante la ejecucion, para calcular los retornos, y actualizar la red de los actores.

        El entrenamiento de la red ahora se ejecuta en cada iteración.

# Cómo Usar:

    Ejecuta el código.

    Ingresa comandos de texto.

    El agente ahora usará el algoritmo A2C para aprender a tomar mejores acciones en el entorno simulado.

    Haz clic en "Entrenar" para entrenar el agente.

# Puntos Clave de la Integración A2C:

    Política y Valor: A2C entrena tanto la política (cómo actuar) como el valor (qué tan bueno es estar en un estado), mejorando la estabilidad del aprendizaje.

    Ventaja: La ventaja ayuda a reducir la varianza en el entrenamiento, haciendo que la convergencia sea más rápida y más estable.

    Recompensas acumuladas: A2C actualiza las redes utilizando el retorno (recompensa acumulada) en lugar de solo la recompensa inmediata.

    Actualizacion de redes por cada iteracion: En este caso, las redes del actor y el critico se actualizan en cada iteración, con el objetivo de converger a un agente que tome las mejores acciones.

# Mejoras y Expansiones:

    Arquitectura de la Red: Experimentar con diferentes arquitecturas de redes neuronales (más capas, diferentes funciones de activación).

    Hiperparámetros: Ajustar los hiperparámetros como la tasa de aprendizaje, gamma, etc.

    Visualizaciones: Añadir visualizaciones de las curvas de aprendizaje del agente.

    Entorno mas complejo: Agregar más objetos y acciones al entorno para una mayor complejidad."""

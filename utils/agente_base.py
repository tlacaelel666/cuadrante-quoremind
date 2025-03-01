"""
This script defines a base class for different types of folder and their respective subclasses.
Each agent type demonstrates a specific form of machine learning approach: supervised learning
and reinforcement learning. The script also includes an example execution to show interactions
and learning behavior of the folder.
"""

import ast
import random

class AgentBase:
    """
    Base class for folder, providing shared mechanics and common functionalities.

    Attributes:
        name (str): The name of the agent.
    """
    def __init__(self, name: str):
        self.name = name
        """
    Defines how to interact with nq_a and nq_b by parsing their string configurations.

    Parameters:
        quantum_state (str): The quantum state as a string to be parsed and interacted with.
        nq_b (str): The string containing information about quantum state nq_b.
        nq_a (str): The string containing information about quantum state nq_a.

    Returns:
        str: A processed message describing the interaction result.

    Raises:
        ValueError: If `quantum_state` or `nq_a` cannot be successfully parsed.
        NotImplementedError: If `nq_a` does not contain `nq_b` data.
    """
    def interact_with_nqa(self, quantum_state: str, nq_b: str, nq_a: str) -> str:
        self.interact_with_nqb(quantum_state, nq_b)
        return self.interact_with_nqb(quantum_state, nq_a)

    def interact_with_nqb(self, quantum_state: str, nq_a: str) -> str:
        self.interact_with_nqb(quantum_state, nq_a)
        return self.interact_with_nqb(quantum_state, nq_a)

    def _parse_input(self, quantum_state: str, nq_a: str):
        """Safely parses the input strings using ast.literal_eval."""
        try:
            parsed_state = ast.literal_eval(quantum_state)  # Safely parse quantum_state
            parsed_nq_a = ast.literal_eval(nq_a)  # Safely parse nq_a
            if 'nq_b' in parsed_nq_a:
                nq_b_data = parsed_nq_a['nq_b']
                return parsed_state, nq_b_data
            else:
                raise NotImplementedError("This method requires 'nq_b' in nq_a to operate.")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse input data: {e}")


class SupervisedAgent(AgentBase):
    """
    Represents an agent that learns using supervised learning techniques. This agent
    processes quantum states and learns from training data to make informed decisions.

    Attributes:
        training_data (list): A list to simulate stored training data for the agent.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.training_data = []  # Simular datos para el entrenamiento supervisado.

    def interact_with_nqb(self, quantum_state: str, nq_b: str) -> str:
        """
        Interacts with the given quantum state and processes `nq_a` information.
        
        Parameters:
            quantum_state (str): The quantum state as a string, expected to evaluate to a dictionary.
            nq_b (str): An additional parameter representing another quantum state, reserved for future extensions.
        
        Returns:
            str: A message indicating whether the `nq_a` configuration was successfully processed.
        """
        # Parseamos las configuraciones:
        nq_a = ast.literal_eval(quantum_state)  # {'nq_a': '1:0', '0': '-1'}
        if 'nq_a' in nq_a:
            result = f"Supervised agent {self.name} processes nq_a: {nq_a['nq_a']}"
        else:
            result = f"Supervised agent {self.name} doesn't understand nq_a."

        print(result)
        return result

    def learn(self) -> None:
        """
        Simulates supervised learning by processing stored training data.
        This method demonstrates how the agent updates its internal model.
        """
        # Simulamos una actualización de parámetros con datos de entrenamiento.
        print(f"Supervised agent {self.name} is learning from training data...")
        # Ejemplo simple, cargar y procesar algún set:
        self.training_data.append({"sample": "nq_a", "output": "prediction"})
        """
        Simulates interactions with both `nq_a` and `nq_b` quantum states.
        This method is designed to show the sequence of interactions with different quantum configurations.
        """
    def interact_with_nq(self, quantum_state, nq_a, nq_b):
        self.interact_with_nqb(quantum_state, nq_b)
        self.interact_with_nqa(quantum_state, nq_a, nq_b)
        self.learn()


class ReinforcementAgent(AgentBase):

    """
    Represents an agent that learns using reinforcement learning techniques. This agent
    interacts with quantum states to maximize its cumulative reward by exploring and exploiting.

    Attributes:
        rewards (int): Tracks the cumulative rewards earned by the agent.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.rewards = 0  # Mantiene la recompensa acumulada.

    def interact_with_nq(self, quantum_state: str, nq_b: str) -> str:
        quantum_state.join(self.interact_with_nqb(quantum_state, nq_b))
        """
        Interacts with the quantum state `nq_b` using reinforcement learning techniques.
        
        Parameters:
            quantum_state (str): The quantum state as a string to be parsed and processed.
            nq_b (str): The string containing information about `nq_b` quantum configuration.
        
        Returns:
            str: A message describing the interaction result with `nq_b`.
        """
        try:
            nq_b = ast.literal_eval(nq_b)
            if 'nq_b' in nq_b:
                result = f"Reinforcement agent {self.name} processes nq_b: {nq_b['nq_b']}"
            elif 'nq_a' in nq_b:
                result = f"Reinforcement agent {self.name} doesn't understand nq_b."
            else: result = "No data found"
        except (ValueError, SyntaxError) as e: result = f"Error parsing nq_b: {e}"
        print(result)
        return result

    def learn(self) -> None:
        """
        Simulates reinforcement learning by selecting random actions (explore or exploit)
        and updating the agent's cumulative rewards based on the received outcomes.
        """
        # Simular un sistema para maximizar una recompensa.
        action = random.choice(["explore", "exploit"])  # Elegir acciones aleatorias.
        if action == "explore":
            reward = random.randint(-1, 1)  # Explorar puede ser más riesgoso.
        else:
            reward = 1  # Explotar es seguro y da recompensa positiva.

        self.rewards += reward
        print(f"Reinforcement agent {self.name}: Action: {action}, Reward: {reward}, Total Rewards: {self.rewards}")


# Example execution: Create folder and simulate their interactions with quantum states `nq_a` and `nq_b`.

def main():
    """
    Main function to demonstrate the functionality of supervised and reinforcement folder.
    Creates instances of both folder, performs quantum state interactions, and simulates learning processes.
    """
    # Configuraciones (tomadas de `State`).
    quantum_state = "{'nq_a':'1:0', '0':'-1'}"
    nq_a = "{'nq_b':'0:-1', '1':'0'}"

    # Crear agentes
    agent1 = SupervisedAgent(name="Agent1")
    agent2 = ReinforcementAgent(name="Agent2")

    # Interacción con los estados
    print("\n** Interaction Phase **")
    print("\n** Using Reinforcement Agent **")
    agent1.interact_with_nq(quantum_state, nq_a=nq_a, nq_b=nq_a)
    agent2.interact_with_nq(quantum_state, nq_b=nq_a)

    # Aprendizaje
    print("\n** Learning Phase **")
    agent1.learn()
    agent2.learn()
"""
This script defines a base class for different types of agents and their respective subclasses.
Each agent type demonstrates a specific form of machine learning approach: supervised learning
and reinforcement learning. The script also includes an example execution to show interactions
and learning behavior of the agents.
"""

import ast
import random
from typing import Dict, Any, Tuple, Optional


class AgentBase:
    """
    Base class for agents, providing shared mechanics and common functionalities.

    Attributes:
        name (str): The name of the agent.
    """
    def __init__(self, name: str):
        self.name = name

    def interact_with_nqa(self, quantum_state: str, nq_b: str, nq_a: str) -> str:
        """
        Defines how to interact with nq_a and nq_b by parsing their string configurations.

        Parameters:
            quantum_state (str): The quantum state as a string to be parsed and interacted with.
            nq_b (str): The string containing information about quantum state nq_b.
            nq_a (str): The string containing information about quantum state nq_a.

        Returns:
            str: A processed message describing the interaction result.
        """
        # First interact with nq_b, then with nq_a
        self.interact_with_nqb(quantum_state, nq_b)
        return self.interact_with_nqb(quantum_state, nq_a)

    def interact_with_nqb(self, quantum_state: str, nq_b: str) -> str:
        """
        Interacts with the given quantum state and processes nq_b information.
        
        Parameters:
            quantum_state (str): The quantum state as a string.
            nq_b (str): The string containing information about quantum state nq_b.
        
        Returns:
            str: A message describing the interaction result.
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement interact_with_nqb")

    def _parse_input(self, quantum_state: str, nq_a: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Safely parses the input strings using ast.literal_eval.
        
        Parameters:
            quantum_state (str): The quantum state as a string.
            nq_a (str): The string containing information about quantum state nq_a.
            
        Returns:
            Tuple[Dict, Dict]: A tuple containing the parsed quantum state and nq_b data.
            
        Raises:
            ValueError: If the input strings cannot be successfully parsed.
            NotImplementedError: If 'nq_b' is not found in nq_a.
        """
        try:
            parsed_state = ast.literal_eval(quantum_state)
            parsed_nq_a = ast.literal_eval(nq_a)
            
            if 'nq_b' in parsed_nq_a:
                nq_b_data = parsed_nq_a['nq_b']
                return parsed_state, parsed_nq_a
            else:
                raise NotImplementedError("This method requires 'nq_b' in nq_a to operate.")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse input data: {e}")
    
    def learn(self) -> None:
        """
        Base learning method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement learn method")


class SupervisedAgent(AgentBase):
    """
    Represents an agent that learns using supervised learning techniques. This agent
    processes quantum states and learns from training data to make informed decisions.

    Attributes:
        training_data (list): A list to simulate stored training data for the agent.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.training_data = []  # Simulate data for supervised training

    def interact_with_nqb(self, quantum_state: str, nq_b: str) -> str:
        """
        Interacts with the given quantum state and processes nq_b information.
        
        Parameters:
            quantum_state (str): The quantum state as a string.
            nq_b (str): The string containing information about quantum state nq_b.
        
        Returns:
            str: A message indicating whether the nq_b configuration was successfully processed.
        """
        try:
            # Parse the configuration
            parsed_state = ast.literal_eval(quantum_state)
            
            if 'nq_a' in parsed_state:
                result = f"Supervised agent {self.name} processes nq_a: {parsed_state['nq_a']}"
            else:
                result = f"Supervised agent {self.name} doesn't understand the quantum state."
                
        except (ValueError, SyntaxError) as e:
            result = f"Error parsing quantum state: {e}"
            
        print(result)
        return result

    def learn(self) -> None:
        """
        Simulates supervised learning by processing stored training data.
        This method demonstrates how the agent updates its internal model.
        """
        print(f"Supervised agent {self.name} is learning from training data...")
        # Simple example, load and process a dataset:
        self.training_data.append({"sample": "nq_a", "output": "prediction"})
        print(f"Updated training data. Total samples: {len(self.training_data)}")

    def interact_with_nq(self, quantum_state: str, nq_a: str, nq_b: str) -> None:
        """
        Simulates interactions with both nq_a and nq_b quantum states.
        
        Parameters:
            quantum_state (str): The quantum state as a string.
            nq_a (str): The string containing information about quantum state nq_a.
            nq_b (str): The string containing information about quantum state nq_b.
        """
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
        self.rewards = 0  # Track accumulated rewards

    def interact_with_nqb(self, quantum_state: str, nq_b: str) -> str:
        """
        Interacts with the quantum state nq_b using reinforcement learning techniques.
        
        Parameters:
            quantum_state (str): The quantum state as a string.
            nq_b (str): The string containing information about nq_b quantum configuration.
        
        Returns:
            str: A message describing the interaction result with nq_b.
        """
        try:
            parsed_nq_b = ast.literal_eval(nq_b)
            if 'nq_b' in parsed_nq_b:
                result = f"Reinforcement agent {self.name} processes nq_b: {parsed_nq_b['nq_b']}"
            elif 'nq_a' in parsed_nq_b:
                result = f"Reinforcement agent {self.name} found nq_a but expected nq_b."
            else: 
                result = f"Reinforcement agent {self.name} found no expected data in: {parsed_nq_b}"
        except (ValueError, SyntaxError) as e: 
            result = f"Error parsing nq_b: {e}"
            
        print(result)
        return result

    def interact_with_nq(self, quantum_state: str, nq_b: str) -> None:
        """
        Main interaction method for the reinforcement agent.
        
        Parameters:
            quantum_state (str): The quantum state as a string.
            nq_b (str): The string containing information about nq_b.
        """
        result = self.interact_with_nqb(quantum_state, nq_b)
        # Learn from this interaction
        self.learn()
        return result

    def learn(self) -> None:
        """
        Simulates reinforcement learning by selecting random actions (explore or exploit)
        and updating the agent's cumulative rewards based on the received outcomes.
        """
        # Simulate a system to maximize a reward
        action = random.choice(["explore", "exploit"])
        if action == "explore":
            reward = random.randint(-1, 1)  # Exploration is risky
        else:
            reward = 1  # Exploitation is safe with positive reward

        self.rewards += reward
        print(f"Reinforcement agent {self.name}: Action: {action}, Reward: {reward}, Total Rewards: {self.rewards}")


def main():
    """
    Main function to demonstrate the functionality of supervised and reinforcement agents.
    Creates instances of both agents, performs quantum state interactions, and simulates learning processes.
    """
    # Configurations
    quantum_state = "{'nq_a':'1:0', '0':'-1'}"
    nq_a = "{'nq_b':'0:-1', '1':'0'}"

    # Create agents
    agent1 = SupervisedAgent(name="Agent1")
    agent2 = ReinforcementAgent(name="Agent2")

    # Interaction with states
    print("\n** Interaction Phase **")
    
    print("\n** Using Supervised Agent **")
    agent1.interact_with_nq(quantum_state, nq_a=nq_a, nq_b=nq_a)
    
    print("\n** Using Reinforcement Agent **")
    agent2.interact_with_nq(quantum_state, nq_b=nq_a)

    # Learning
    print("\n** Additional Learning Phase **")
    agent1.learn()
    agent2.learn()


if __name__ == "__main__":
    main()
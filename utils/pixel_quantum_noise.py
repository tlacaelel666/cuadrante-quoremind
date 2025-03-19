import pygame
import random
import numpy as np
from quantum_bayes_mahalanobis import QuantumNoiseCollapse

class QuantumPixelVisualizer:
    def __init__(self, width=800, height=600, fps=144):
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Quantum Pixel Visualizer")
        
        self.clock = pygame.time.Clock()
        self.fps = fps
        
        # Inicializar el modelo cuántico
        self.quantum_noise_collapse = QuantumNoiseCollapse()
        
        # Estados cuánticos iniciales para generación de píxeles
        self.quantum_states = np.random.rand(3, 2)
        
        # Estado objetivo para optimización
        self.target_state = np.array([1.0, 0.0])
    
    def generate_quantum_pixels(self):
        """
        Genera píxeles usando el modelo cuántico de colapso de onda
        """
        # Optimizar estados cuánticos
        optimized_states, _ = self.quantum_noise_collapse.optimize_quantum_state(
            self.quantum_states, 
            self.target_state,
            max_iterations=50,
            learning_rate=0.01
        )
        
        # Simular colapso de onda
        collapse_result = self.quantum_noise_collapse.simulate_wave_collapse(
            optimized_states,
            prn_influence=0.5,
            previous_action=0
        )
        
        # Crear superficie de píxeles basada en el colapso
        surface = pygame.Surface((self.width, self.height))
        pixels_array = pygame.surfarray.pixels3d(surface)
        
        # Usar información del colapso para generar píxeles
        entropy = collapse_result['entropy']
        coherence = collapse_result['coherence']
        mahalanobis_dist = collapse_result['mahalanobis_distance']
        cosines = collapse_result['cosines']
        
        # Generar píxeles con base en parámetros cuánticos
        r = int((cosines[0] + 1) * 127.5)
        g = int((cosines[1] + 1) * 127.5)
        b = int((cosines[2] + 1) * 127.5)
        
        # Aplicar variación basada en entropía y distancia de Mahalanobis
        noise_factor = np.clip(mahalanobis_dist * entropy, 0, 1)
        r = int(r * (1 + noise_factor))
        g = int(g * (1 + noise_factor))
        b = int(b * (1 + noise_factor))
        
        pixels_array[:] = np.random.randint(0, 256, (self.width, self.height, 3))
        pixels_array[:, :, 0] = np.clip(pixels_array[:, :, 0] * r / 255, 0, 255)
        pixels_array[:, :, 1] = np.clip(pixels_array[:, :, 1] * g / 255, 0, 255)
        pixels_array[:, :, 2] = np.clip(pixels_array[:, :, 2] * b / 255, 0, 255)
        
        del pixels_array
        return surface
    
    def run(self):
        running = True
        frame_count = 0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Generar superficie de píxeles cuánticos
            surface = self.generate_quantum_pixels()
            
            # Dibujar superficie
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            
            # Actualizar estados cuánticos periódicamente
            frame_count += 1
            if frame_count >= 100:
                self.quantum_states = np.random.rand(3, 2)
                frame_count = 0
            
            # Control de FPS
            self.clock.tick(self.fps)
        
        pygame.quit()

# Ejecutar visualizador
if __name__ == "__main__":
    visualizer = QuantumPixelVisualizer()
    visualizer.run()
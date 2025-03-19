import pygame
import random
import numpy as np

class PixelEffectVisualizer:
    def __init__(self, width=800, height=600, fps=144):
        # Inicialización de Pygame
        pygame.init()
        
        # Configuraciones de pantalla
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pixel Effect Visualizer")
        
        # Configuraciones de renderizado
        self.clock = pygame.time.Clock()
        self.fps = fps
        
        # Modos de generación de píxeles
        self.pixel_modes = [
            self._generate_random_pixels,
            self._generate_gradient_pixels,
            self._generate_noise_pixels
        ]
        
        # Contadores y temporizadores
        self.frame_count = 0
        self.mode_change_interval = 300  # Cambiar modo cada 300 frames
    
    def _generate_random_pixels(self):
        """Genera píxeles completamente aleatorios"""
        surface = pygame.Surface((self.width, self.height))
        pixels_array = pygame.surfarray.pixels3d(surface)
        pixels_array[:] = np.random.randint(0, 256, (self.width, self.height, 3))
        del pixels_array
        return surface
    
    def _generate_gradient_pixels(self):
        """Genera un efecto de gradiente suave"""
        surface = pygame.Surface((self.width, self.height))
        pixels_array = pygame.surfarray.pixels3d(surface)
        
        # Gradiente con variación
        r_grad = np.linspace(0, 255, self.width)[:, np.newaxis]
        g_grad = np.linspace(0, 255, self.height)[np.newaxis, :]
        b_grad = np.random.randint(0, 256, (self.width, self.height))
        
        pixels_array[:, :, 0] = r_grad
        pixels_array[:, :, 1] = g_grad.T
        pixels_array[:, :, 2] = b_grad
        
        del pixels_array
        return surface
    
    def _generate_noise_pixels(self):
        """Genera un efecto de ruido más sofisticado"""
        surface = pygame.Surface((self.width, self.height))
        pixels_array = pygame.surfarray.pixels3d(surface)
        
        # Ruido con suavizado
        noise = np.random.normal(128, 50, (self.width, self.height, 3))
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        pixels_array[:] = noise
        
        del pixels_array
        return surface
    
    def run(self):
        """Bucle principal de ejecución"""
        running = True
        current_mode_index = 0
        
        while running:
            # Manejo de eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Cambio manual de modo con teclas
                    if event.key == pygame.K_SPACE:
                        current_mode_index = (current_mode_index + 1) % len(self.pixel_modes)
            
            # Generar píxeles con el modo actual
            surface = self.pixel_modes[current_mode_index]()
            
            # Dibujar superficie
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            
            # Cambio automático de modo
            self.frame_count += 1
            if self.frame_count >= self.mode_change_interval:
                current_mode_index = (current_mode_index + 1) % len(self.pixel_modes)
                self.frame_count = 0
            
            # Control de FPS
            self.clock.tick(self.fps)
        
        pygame.quit()

# Ejecutar visualizador
if __name__ == "__main__":
    visualizer = PixelEffectVisualizer()
    visualizer.run()
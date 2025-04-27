import numpy as np

# Estado colapsado |ψ⟩
psi = np.array([[0.8 + 0.6j], [0.6 - 0.8j]])

# Normalización y ángulos de Bloch
theta = 2 * np.arctan(np.abs(psi[1]) / np.abs(psi[0]))
phi = np.angle(psi[1])

# Matriz unitaria aproximada
U = np.array([
    [np.cos(theta / 2), -np.sin(theta / 2) * np.exp(-1j * phi)],
    [np.sin(theta / 2) * np.exp(1j * phi), np.cos(theta / 2)]
])

# Estado cuántico (vector de amplitudes)
psi = np.array([0.8 + 0.6j, 0.6 - 0.8j])

# Aplicar FFT
fft_psi = np.fft.fft(psi)

# Inversión con IFFT para recuperar el estado original
recovered_psi = np.fft.ifft(fft_psi)

print("FFT del estado:", fft_psi)
print("Estado recuperado con IFFT:", recovered_psi)
print("Matriz Unitaria reconstruida:")
print(U)

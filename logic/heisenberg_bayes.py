import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Constantes físicas
h_bar = 1.0545718e-34  # Constante de Planck reducida (J·s)

def heisenberg_uncertainty(delta_x, delta_p):
    """
    Calcula el producto de incertidumbres y verifica si cumple el principio de Heisenberg
    """
    product = delta_x * delta_p
    limit = h_bar / 2
    satisfies = product >= limit
    
    return {
        "producto": product,
        "limite": limit,
        "cumple_desigualdad": satisfies
    }

def bayesian_update(prior, likelihood, evidence):
    """
    Actualiza una distribución de probabilidad utilizando el teorema de Bayes
    prior: p(x) - distribución de probabilidad inicial
    likelihood: p(y|x) - probabilidad de la observación dada la hipótesis
    evidence: p(y) - probabilidad marginal de la observación
    """
    posterior = (likelihood * prior) / evidence
    return posterior

def shannon_entropy(probabilities):
    """
    Calcula la entropía de Shannon para una distribución de probabilidad
    """
    # Filtrar probabilidades cero para evitar log(0)
    valid_probs = probabilities[probabilities > 0]
    return -np.sum(valid_probs * np.log2(valid_probs))

def conditional_entropy(joint_prob, marginal_prob):
    """
    Calcula la entropía condicional H(X|Y)
    """
    # Calcula p(x|y) = p(x,y)/p(y)
    conditional_prob = joint_prob / marginal_prob[:, np.newaxis]
    
    # H(X|Y) = -Σ p(y) Σ p(x|y) log p(x|y)
    h_cond = 0
    for i in range(len(marginal_prob)):
        entropy_given_y = -np.sum(conditional_prob[i] * np.log2(conditional_prob[i] + 1e-10))
        h_cond += marginal_prob[i] * entropy_given_y
    
    return h_cond

def gaussian_pdf(x, mu, sigma):
    """Función de densidad de probabilidad gaussiana"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def joint_gaussian_distribution(x_grid, p_grid, delta_x, delta_p, rho=0):
    """
    Crea una distribución de probabilidad conjunta gaussiana para posición (x) y momento (p)
    con un coeficiente de correlación rho
    """
    n = len(x_grid)
    joint_prob = np.zeros((n, n))
    
    for i, x in enumerate(x_grid):
        for j, p in enumerate(p_grid):
            # Fórmula de la distribución normal bivariada
            z = ((x/delta_x)**2 - 2*rho*(x/delta_x)*(p/delta_p) + (p/delta_p)**2) / (1 - rho**2)
            joint_prob[i, j] = np.exp(-z/2) / (2*np.pi*delta_x*delta_p*np.sqrt(1-rho**2))
            
    # Normalizar para asegurar que suma 1
    joint_prob = joint_prob / np.sum(joint_prob)
    return joint_prob

def confidence_ellipse(x_std, y_std, ax, n_std=3.0, **kwargs):
    """
    Crea una elipse de confianza para representar la incertidumbre
    """
    ellipse = Ellipse((0, 0), width=x_std*2*n_std, height=y_std*2*n_std, **kwargs)
    
    # Transformación
    transf = transforms.Affine2D().rotate_deg(0) + ax.transData
    ellipse.set_transform(transf)
    
    return ax.add_patch(ellipse)

# Demostración de las relaciones
if __name__ == "__main__":
    print("Explorando la conexión entre Heisenberg, Bayes y Entropía de Shannon")
    print("-----------------------------------------------------------------")
    
    # 1. Demostración del principio de incertidumbre de Heisenberg
    print("\n1. PRINCIPIO DE INCERTIDUMBRE DE HEISENBERG")
    
    # Unidades simplificadas (ħ = 1)
    h_bar_simplified = 1.0
    
    # Caso con mínima incertidumbre (estado coherente)
    delta_x_min = 1.0
    delta_p_min = h_bar_simplified / (2 * delta_x_min)
    
    result_min = heisenberg_uncertainty(delta_x_min, delta_p_min)
    print(f"Estado de mínima incertidumbre:")
    print(f"  Δx = {delta_x_min}, Δp = {delta_p_min}")
    print(f"  Producto Δx·Δp = {result_min['producto']}")
    print(f"  Límite ħ/2 = {result_min['limite']}")
    print(f"  Cumple la desigualdad: {result_min['cumple_desigualdad']}")
    
    # Caso con mayor incertidumbre
    delta_x_large = 2.0
    delta_p_large = h_bar_simplified
    
    result_large = heisenberg_uncertainty(delta_x_large, delta_p_large)
    print(f"\nEstado con mayor incertidumbre:")
    print(f"  Δx = {delta_x_large}, Δp = {delta_p_large}")
    print(f"  Producto Δx·Δp = {result_large['producto']}")
    print(f"  Límite ħ/2 = {result_large['limite']}")
    print(f"  Cumple la desigualdad: {result_large['cumple_desigualdad']}")
    
    # 2. Visualización de la densidad de probabilidad conjunta
    print("\n2. DENSIDAD DE PROBABILIDAD CONJUNTA")
    
    # Crear una malla para x y p
    x = np.linspace(-5, 5, 100)
    p = np.linspace(-5, 5, 100)
    
    # Distribución conjunta para el estado de mínima incertidumbre
    joint_prob = joint_gaussian_distribution(x, p, delta_x_min, delta_p_min)
    
    # Calcular las distribuciones marginales
    p_x = np.sum(joint_prob, axis=1)  # Marginalizando sobre p
    p_p = np.sum(joint_prob, axis=0)  # Marginalizando sobre x
    
    # Entropía de las distribuciones marginales
    H_x = shannon_entropy(p_x)
    H_p = shannon_entropy(p_p)
    
    print(f"Entropía de Shannon para posición H(X): {H_x:.4f} bits")
    print(f"Entropía de Shannon para momento H(P): {H_p:.4f} bits")
    print(f"Suma de entropías H(X) + H(P): {H_x + H_p:.4f} bits")
    
    # 3. Actualización bayesiana
    print("\n3. ACTUALIZACIÓN BAYESIANA")
    
    # Supongamos que tenemos una distribución previa para la posición
    prior_x = p_x  # Usando la distribución marginal de x como prior
    
    # Y realizamos una medición con una cierta precisión (likelihood)
    measurement_pos = 1.0  # Posición medida
    measurement_uncertainty = 0.5  # Incertidumbre en la medición
    
    likelihood = gaussian_pdf(x, measurement_pos, measurement_uncertainty)
    
    # La evidencia es la probabilidad marginal de la observación
    evidence = np.sum(likelihood * prior_x)
    
    # Actualización bayesiana
    posterior_x = bayesian_update(prior_x, likelihood, evidence)
    
    # Calcular la entropía del posterior
    H_x_posterior = shannon_entropy(posterior_x)
    print(f"Entropía del prior H(X): {H_x:.4f} bits")
    print(f"Entropía del posterior H(X|Y): {H_x_posterior:.4f} bits")
    print(f"Reducción de entropía: {H_x - H_x_posterior:.4f} bits")
    
    # 4. Conexión con la relación de incertidumbre
    print("\n4. CONEXIÓN ENTRE INCERTIDUMBRE Y ENTROPÍA")
    
    # Relación información-incertidumbre
    print("La desigualdad de Heisenberg establece un límite fundamental")
    print("para el producto de incertidumbres: Δx·Δp ≥ ħ/2")
    print("\nEsto se relaciona con la teoría de la información:")
    print("- Mayor entropía implica mayor incertidumbre")
    print("- La actualización bayesiana reduce la entropía (incertidumbre)")
    print("- Existe un límite mínimo de incertidumbre cuántica")
    
    # Visualización de resultados
    plt.figure(figsize=(15, 10))
    
    # 1. Distribución conjunta y principio de incertidumbre
    ax1 = plt.subplot(2, 2, 1)
    plt.imshow(joint_prob, extent=[-5, 5, -5, 5], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Probabilidad')
    
    # Añadir elipse de incertidumbre
    confidence_ellipse(delta_x_min, delta_p_min, ax1, n_std=1, edgecolor='red', facecolor='none', label='Δx·Δp límite')
    
    plt.title('Distribución de probabilidad conjunta posición-momento')
    plt.xlabel('Posición (x)')
    plt.ylabel('Momento (p)')
    
    # 2. Distribuciones marginales
    ax2 = plt.subplot(2, 2, 2)
    plt.plot(x, p_x, 'b-', label=f'p(x), H(X)={H_x:.2f} bits')
    plt.plot(p, p_p, 'r-', label=f'p(p), H(P)={H_p:.2f} bits')
    plt.title('Distribuciones marginales y entropías')
    plt.xlabel('Valor')
    plt.ylabel('Probabilidad')
    plt.legend()
    
    # 3. Actualización bayesiana
    ax3 = plt.subplot(2, 2, 3)
    plt.plot(x, prior_x, 'b-', label=f'Prior p(x), H={H_x:.2f}')
    plt.plot(x, likelihood, 'g-', label=f'Likelihood p(y|x)')
    plt.plot(x, posterior_x, 'r-', label=f'Posterior p(x|y), H={H_x_posterior:.2f}')
    plt.axvline(x=measurement_pos, color='k', linestyle='--', label='Medición')
    plt.title('Actualización Bayesiana y reducción de entropía')
    plt.xlabel('Posición (x)')
    plt.ylabel('Probabilidad')
    plt.legend()
    
    # 4. Relación entre entropía y la desigualdad de Heisenberg
    ax4 = plt.subplot(2, 2, 4)
    
    # Definimos diferentes valores de delta_x y calculamos los delta_p mínimos correspondientes
    delta_x_values = np.linspace(0.5, 3, 100)
    delta_p_values = h_bar_simplified / (2 * delta_x_values)
    
    # Calculamos la entropía para distribuciones gaussianas con estas desviaciones
    entropy_x = np.log2(delta_x_values * np.sqrt(2 * np.pi * np.e))
    entropy_p = np.log2(delta_p_values * np.sqrt(2 * np.pi * np.e))
    
    plt.plot(delta_x_values, entropy_x + entropy_p, 'b-', label='Suma de entropías')
    plt.axhline(y=np.log2(np.pi * np.e * h_bar_simplified), color='r', linestyle='--', 
                label='Límite inf. teórico')
    
    plt.title('Relación entre entropía y principio de incertidumbre')
    plt.xlabel('Δx')
    plt.ylabel('H(X) + H(P) [bits]')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('heisenberg_bayes_shannon.png', dpi=300)
    print("\nFigura guardada como 'heisenberg_bayes_shannon.png'")

    # Conclusiones
    print("\nCONCLUSIONES:")
    print("1. El principio de incertidumbre de Heisenberg establece un límite")
    print("   cuántico fundamental en la precisión con que podemos conocer")
    print("   simultáneamente variables complementarias como posición y momento.")
    print("\n2. Este límite está relacionado con la entropía de Shannon, que")
    print("   cuantifica la incertidumbre en términos de información.")
    print("\n3. El teorema de Bayes permite actualizar nuestro conocimiento")
    print("   (reducir la entropía) a partir de nuevas observaciones.")
    print("\n4. Sin embargo, la mecánica cuántica establece que existe un límite")
    print("   mínimo a la incertidumbre total, expresado tanto en el")
    print("   principio de Heisenberg como en un límite mínimo para la")
    print("   suma de entropías de variables complementarias.")
"""A continuación se muestra un ejemplo de tests unitarios con Pytest para verificar distintas partes de bayes_logic.py. Suponiendo que el archivo se llama bayes_logic.py y se encuentra en el mismo directorio, podrías crear un archivo nuevo (por ejemplo, test_bayes_logic.py) con el siguiente contenido:

--------------------------------------------------------------------------------"""
# test_bayes_logic.py

import pytest
import numpy as np
import tensorflow as tf

from bayes_logic import (
    BayesLogic,
    shannon_entropy,
    calculate_cosines,
    calculate_covariance_matrix,
    calculate_covariance_between_two_variables,
    compute_mahalanobis_distance,
    PRN,
)


@pytest.mark.parametrize("prior_a, prior_b, cond_b_given_a, expected", [
    (0.5, 0.5, 0.5, 0.5),  # Caso balanceado
    (0.8, 0.4, 0.5, 1.0),  # Caso donde posterior = (0.5*0.8)/0.4 = 1.0
    (0.2, 0.8, 0.2, 0.05), # (0.2*0.2)/0.8 = 0.05
])
def test_calculate_posterior_probability(prior_a, prior_b, cond_b_given_a, expected):
    bayes = BayesLogic()
    result = bayes.calculate_posterior_probability(prior_a, prior_b, cond_b_given_a)
    assert pytest.approx(result, 0.001) == expected


@pytest.mark.parametrize("joint_prob, prior, expected", [
    (0.4, 0.5, 0.8),
    (0.2, 0.4, 0.5),
    (0.0, 0.3, 0.0),
])
def test_calculate_conditional_probability(joint_prob, prior, expected):
    bayes = BayesLogic()
    result = bayes.calculate_conditional_probability(joint_prob, prior)
    assert pytest.approx(result, 0.001) == expected


@pytest.mark.parametrize("entropy, expected", [
    (0.9, 0.3),  # Entropía alta
    (0.3, 0.1),  # Entropía baja
])
def test_calculate_high_entropy_prior(entropy, expected):
    bayes = BayesLogic()
    result = bayes.calculate_high_entropy_prior(entropy)
    assert result == expected


@pytest.mark.parametrize("coherence, expected", [
    (0.7, 0.6),  # Coherencia alta
    (0.4, 0.2),  # Coherencia baja
])
def test_calculate_high_coherence_prior(coherence, expected):
    bayes = BayesLogic()
    result = bayes.calculate_high_coherence_prior(coherence)
    assert result == expected


@pytest.mark.parametrize("coherence, action, prn_influence, expected_range", [
    # Se verifican rangos en lugar de un valor exacto, dado que hay cálculo condicional
    (0.7, 1, 0.8, (0.0, 1.0)),  # Alta coherencia, acción=1, PRN alto
    (0.7, 0, 0.2, (0.0, 1.0)),  # Alta coherencia, acción=0, PRN bajo
    (0.5, 1, 0.5, (0.3, 0.3)),  # Coherencia baja => 0.3 fijo
])
def test_calculate_joint_probability(coherence, action, prn_influence, expected_range):
    """
    En el último caso se compara que el resultado sea exactamente 0.3 para coherencia < threshold.
    """
    bayes = BayesLogic()
    result = bayes.calculate_joint_probability(coherence, action, prn_influence)
    if expected_range[0] == expected_range[1]:
        # Si se espera un valor exacto
        assert result == expected_range[0]
    else:
        # Si se espera un rango [low, high]
        low, high = expected_range
        assert low <= result <= high


def test_calculate_probabilities_and_select_action():
    bayes = BayesLogic()
    # Entropía alta y coherencia alta tienden a elevar la prob. de acción
    result = bayes.calculate_probabilities_and_select_action(
        entropy=0.9, coherence=0.7, prn_influence=0.8, action=1
    )
    # Verificamos que las llaves existan
    assert "action_to_take" in result
    assert "high_entropy_prior" in result
    assert "high_coherence_prior" in result
    assert "posterior_a_given_b" in result
    assert "conditional_action_given_b" in result
    
    # Acción podría ser 1 si la prob condicional pasa el umbral
    # (no necesariamente siempre será 1, se revisa la coherencia interna)
    assert result["action_to_take"] in [0, 1]


def test_shannon_entropy():
    data = [1, 1, 2, 2, 3]
    # Con tres valores, la entropía no será cero ni muy alta.
    ent = shannon_entropy(data)
    assert ent > 0.0


def test_calculate_cosines():
    cos_x, cos_y, cos_z = calculate_cosines(entropy=0.8, prn_object=0.6)
    magnitude = np.sqrt(0.8**2 + 0.6**2 + 1)
    
    # Se verifica que cos_x = 0.8 / magnitude, etc.
    assert pytest.approx(cos_x, 0.0001) == 0.8 / magnitude
    assert pytest.approx(cos_y, 0.0001) == 0.6 / magnitude
    assert pytest.approx(cos_z, 0.0001) == 1 / magnitude


def test_calculate_covariance_matrix():
    # 2 variables, 5 observaciones
    data = tf.constant([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ], dtype=tf.float32)
    cov_matrix = calculate_covariance_matrix(data)
    assert cov_matrix.shape == (2, 2)
    # Covarianza no debería ser cero, pues hay variabilidad
    assert not np.allclose(cov_matrix, 0)


def test_calculate_covariance_between_two_variables():
    data = tf.constant([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
    ], dtype=tf.float32)
    cov_manual, cov_tfp = calculate_covariance_between_two_variables(data)
    # Verificamos que ambas formas de cálculo sean cercanas
    assert pytest.approx(cov_manual, 1e-5) == cov_tfp


def test_compute_mahalanobis_distance():
    data = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [2.0, 2.5],
    ]
    point = [2.5, 2.5]
    # Distancia de Mahalanobis no debería ser negativa
    distance = compute_mahalanobis_distance(data, point)
    assert distance >= 0.0


def test_prn_init():
    # Verifica la inicialización de PRN
    prn_obj = PRN(influence=0.5, algorithm_type='tipo_A', alfa=0.3)
    assert prn_obj.influence == 0.5
    assert prn_obj.algorithm_type == 'tipo_A'
    assert prn_obj.parameters.get("alfa") == 0.3

    # Verifica que lance ValueError si influence está fuera de [0,1]
    with pytest.raises(ValueError):
        PRN(influence=1.5)


def test_prn_adjust_influence():
    prn_obj = PRN(0.5)
    prn_obj.adjust_influence(0.3)
    assert prn_obj.influence == 0.8

    # Si sobrepasa 1, se trunca
    prn_obj.adjust_influence(0.5)
    assert prn_obj.influence == 1.0

    # Si queda por debajo de 0, se trunca
    prn_obj.adjust_influence(-2.0)
    assert prn_obj.influence == 0.0


def test_prn_combine_with():
    prn1 = PRN(influence=0.2, algorithm_type='tipo1', param1=10)
    prn2 = PRN(influence=0.8, algorithm_type='tipo2', param2=20)
    combined = prn1.combine_with(prn2, weight=0.3)

    # La influencia combinada se calcula como: 0.2*0.3 + 0.8*0.7 = 0.62
    assert pytest.approx(combined.influence, 0.0001) == 0.62
    # Como weight=0.3 (< 0.5), se toma 'tipo2'
    assert combined.algorithm_type == 'tipo2'
    # Parámetros deben combinarse en un solo diccionario
    assert 'param1' in combined.parameters
    assert 'param2' in combined.parameters
--------------------------------------------------------------------------------

Instrucciones de uso:  
1. Asegúrate de que el archivo bayes_logic.py y test_bayes_logic.py se encuentren en el mismo directorio (o configura correctamente los import en caso de usar otro layout de proyecto).  
2. Instala pytest si aún no lo tienes: pip install pytest.  
3. Desde la línea de comandos, sitúate en el directorio que contiene ambos archivos y ejecuta:  
   pytest -v  

Esto ejecutará todos los tests marcados con pytest. De esta manera podrás validar que las distintas funciones y clases de bayes_logic.py funcionan correctamente.
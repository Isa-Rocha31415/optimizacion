import numpy as np
def c_armijo(f_k, f_k1, g_k, g_k1, P_k, alpha_k):
    """
    Retorna True si NO se cumple el descenso suficiente.
    """
    c1 = 1e-4
    slope = np.dot(g_k.flatten(), P_k.flatten())
    
    # Condición de Armijo: f(x+ap) <= f(x) + c1 * a * grad^T * p
    sufficient_decrease = f_k1 <= (f_k + c1 * alpha_k * slope)
    
    # Si f_k1 es un array (ej. de un elemento), sufficient_decrease será un array.
    # Forzamos a que sea un booleano simple de Python:
    return bool(np.all(sufficient_decrease))
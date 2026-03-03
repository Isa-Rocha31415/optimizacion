import numpy as np
def c_armijo(f_k,f_k1, g_k, P_k, alpha_k,c1=None):
    """
    Retorna True si NO se cumple el descenso suficiente.
    """
    slope = np.dot(g_k, P_k)
    sufficient_decrease = f_k1 <= f_k + c1 * alpha_k * slope
    return not sufficient_decrease
import numpy as np
def curvatura(f_k, f_k1, g_k, P_k, alpha_k):

    c2 = 0.9
    
    pendiente_actual = np.dot(g_k, P_k)
    pendiente_nueva_aprox = (f_k1 - f_k) / alpha_k
    return pendiente_nueva_aprox < c2 * pendiente_actual
    pass    

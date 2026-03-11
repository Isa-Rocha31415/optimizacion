import numpy as np

def c_goldstein(f_k, f_k1, g_k, g_k1, P_k, alpha_k, c=0.1):
    """
    Retorna True si la condición NO se cumple (para que el backtracking siga reduciendo alpha).
    Retorna False si el alpha es aceptable bajo Goldstein.
    """
    # 1. Calculamos el producto punto entre el gradiente y la dirección de búsqueda
    # Esto es la derivada direccional en el punto x_k
    slope = np.dot(g_k, P_k)
    
    # 2. Condición de suficiente descenso (Armijo/Lado derecho)
    # f(x + ap) <= f(x) + c * a * grad.T * p
    sufficient_decrease = f_k1 <= f_k + c * alpha_k * slope
    
    # 3. Condición de control por abajo (Lado izquierdo)
    # f(x) + (1-c) * a * grad.T * p <= f(x + ap)
    not_too_short = f_k1 >= f_k + (1 - c) * alpha_k * slope
    
    # El alpha es bueno si AMBAS son True. 
    # El bucle while necesita un True para seguir iterando (cuando el alpha es MALO).
    # Por lo tanto, regresamos la negación de (A y B)
    return not (sufficient_decrease and not_too_short)
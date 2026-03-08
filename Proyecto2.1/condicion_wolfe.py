import numpy as np
def c_wolfe(f_k, f_k1, g_k, g_k1, P_k, alpha_k):
    """
    Condición de Wolfe (Armijo + curvatura).

    Armijo: f(x + α·p) ≤ f(x) + c₁·α·∇f(x)ᵀ·p
    Curvatura: ∇f(x + α·p)ᵀ·p ≥ c₂·∇f(x)ᵀ·p

    Si NO se cumple alguna → retorna True → seguir reduciendo/buscando α.
    """
    c1 = 1e-4
    c2 = 0.9

    # Armijo
    if f_k1 > f_k + c1 * alpha_k * np.dot(g_k, P_k):
        return True

    # Curvatura (Wolfe)
    if np.dot(g_k1, P_k) < c2 * np.dot(g_k, P_k):
        return True

    return False
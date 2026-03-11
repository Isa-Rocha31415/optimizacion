import numpy as np
def c_fuertewolfe(f_k, f_k1, g_k, g_k1, P_k, alpha_k):
    """
    Condición de Wolfe Fuerte (Armijo + curvatura fuerte).
    En el contexto de backtracking, se verifica Armijo.

    Armijo: f(x + α·p) ≤ f(x) + c₁·α·∇f(x)ᵀ·p
    Curvatura fuerte: |∇f(x + α·p)ᵀ·p| ≤ c₂·|∇f(x)ᵀ·p|

    Si NO se cumple alguna → retorna True → seguir reduciendo/buscando α.
    """
    c1 = 1e-4
    c2 = 0.9

    gk_pk = np.dot(g_k.flatten(), P_k.flatten())
    gk1_pk = np.dot(g_k1.flatten(), P_k.flatten())

    # Armijo
    if f_k1 > f_k + c1 * alpha_k * gk_pk:
        return True

    # Curvatura fuerte (Strong Wolfe)
    if abs(gk1_pk) > c2 * abs(gk_pk):
        return True

    return False
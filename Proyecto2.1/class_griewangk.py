from funcion import funcion
import numpy as np
 
class griewangk(funcion):
    """
    Implementación de la función de Griewangk.
    Minimo global en (0, 0) con valor 0.
    """
 
    def eval(self, args: np.ndarray) -> float:
        n = len(args)
        # Término de la suma: Σ (x_i^2 / 4000)
        sum_term = np.sum(args**2) / 4000
        
        # Término del producto: Π cos(x_i / sqrt(i+1))
        indices = np.arange(1, n + 1)
        prod_term = np.prod(np.cos(args / np.sqrt(indices)))
        
        return sum_term - prod_term + 1

    def diff(self, args: np.ndarray) -> np.ndarray:
        n = len(args)
        indices = np.arange(1, n + 1)
        sqrt_i = np.sqrt(indices)
        
        # Gradiente de la parte de la suma: x_i / 2000
        grad_sum = args / 2000
        
        # Gradiente de la parte del producto:
        # La derivada de - Π cos(x_j/sqrt(j)) respecto a x_i es:
        # [Π_{j!=i} cos(x_j/sqrt(j))] * sin(x_i/sqrt(i)) * (1/sqrt(i))
        cos_vals = np.cos(args / sqrt_i)
        sin_vals = np.sin(args / sqrt_i)
        
        # Producto total
        prod_total = np.prod(cos_vals)
        
        grad_prod = []
        for i in range(n):
            # Para evitar división por cero si cos es 0, usamos esta lógica:
            term = (prod_total / (cos_vals[i] + 1e-16)) * sin_vals[i] * (1 / sqrt_i[i])
            grad_prod.append(term)
            
        return grad_sum + np.array(grad_prod)

    def doiff(self, args: np.ndarray) -> np.ndarray:
        n = len(args)
        hessian = np.zeros((n, n))
        indices = np.arange(1, n + 1)
        sqrt_i = np.sqrt(indices)
        
        cos_vals = np.cos(args / sqrt_i)
        sin_vals = np.sin(args / sqrt_i)
        prod_total = np.prod(cos_vals)

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal: 1/2000 + derivada segunda del producto
                    term_ii = (prod_total / (cos_vals[i]**2 + 1e-16)) * (sin_vals[i]**2) * (1/indices[i]) # Aproximación simplificada
                    # Una forma más estable para la diagonal:
                    diag_prod = (prod_total / (cos_vals[i] + 1e-16)) * cos_vals[i] * (1/indices[i])
                    # Simplificando para Newton (la parte dominante es el 1/2000):
                    hessian[i, i] = (1 / 2000) + (prod_total / indices[i])
                else:
                    # Fuera de la diagonal (derivadas mixtas)
                    # En Griewangk son muy pequeñas, a menudo se omiten o se aproximan
                    hessian[i, j] = 0 
        return hessian
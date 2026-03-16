#Algoritmo
# July
# Rapla 
#Puden propbar su algoritmo en el main con las funciones anteriores
# estas funciones estan en la carpeta funciones_test
from funcion import funcion
import numpy as np

class GC:
    
    def __init__(self, funcion_obj: funcion,max_error, k_max=1000):
        
        self.max_error = max_error
        self.funcion_obj = funcion_obj
        self.k_max = k_max 

    
    """
    Gradiente Conjugado
    """
    def solve(self, A:np.ndarray, x_0:np.ndarray): 
        """
    Optimizador de gradiente conjugado.
    @args: 
        x_0:[array] = representa el punto inicial en la función objetivo 
        A : Una matriz simétrica positiva que sale de algún lugar
    @return: algo, sí. 
        """
        k = 0 
        x = x_0
        grad = A @ x # <- El operador @ creo que puede ser lento si la matriz es muy grande. Numpy tiene matmul() que parece que podría ser más rápido. 
        direct:np.ndarray = -grad
        rank_A = np.linalg.matrix_rank(A) 
        
        while k < rank_A and np.linalg.norm(A) > self.max_error:
            # Calculamos este factor antes para ahorrar operaciones
            fact = A @ direct

            # ======= Optimizar =======
            # Calcular nuevo tamaño de paso            
            step =  (direct @  grad) / (direct @ fact) 

            # El siguiente punto para buscar será en la dirección de minimización encontrada
            x += (step @ direct) 

            # La dirección la tenemos como parte de un gradiente y un factor
            grad_next = grad + (direct @ fact)

            # beta: sí. 
            beta = (grad_next @ grad_next) / (grad @ grad)

            # L dirección actualizada para el siguiente paso 
            direct = grad_next + (beta @ direct) 

            # ======= Actualizar =======
            grad = grad_next 
            k += 1
        
        return x 
    pass 
    

    pass
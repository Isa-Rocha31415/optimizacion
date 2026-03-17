#Algoritmo
# July
# Rapla 
#Puden propbar su algoritmo en el main con las funciones anteriores
# estas funciones estan en la carpeta funciones_test
from funcion import funcion
import numpy as np

class GC:
    
    def __init__(self, funcion_obj: funcion, k_max=1000):
        
        self.max_error = 1e-6 
        self.funcion_obj = funcion_obj
        self.k_max = k_max 

    
    """
    Gradiente Conjugado
    """
    def solve(self, x_0:np.ndarray): 
        """
    Optimizador de gradiente conjugado.
    @args: 
        x_0:[array] = representa el punto inicial en la función objetivo 
        A : Una matriz simétrica positiva que sale de algún lugar
    @return: 
        x: ndarray = el punto mínimo encontrado 
        """
        k = 0 
        x = x_0.flatten().copy() 

        # g0 = Ax - b  
        grad = self.funcion_obj.diff(x) 

        # p0 = -g0
        direct:np.ndarray = -grad
        rank  = x_0.size
        
        while k < rank and np.linalg.norm(grad) > self.max_error:
            print("Encontro al ciclo")
            # Calculamos este factor antes para ahorrar operaciones
            # Esto es lo de Ap_x pero se está implementado de forma numérica en funcion.diof() 
            fact = self.funcion_obj.doif(direct) 

            # ======= Optimizar =======
            # de: (g_k^T * g_k) / (p_k^T * A * p_k)
            # Sacamos solo el denominador 
            print("optimizar")
            denom = np.dot(direct, fact) # <- Podría dar un división por 0 así que hay que reisar la consola
            
            # Calcular nuevo tamaño de paso            
            step =  np.dot(grad, grad) / denom 

            # El siguiente punto para buscar será en la dirección de minimización encontrada
            x = x + (step * direct) 

            # La dirección la tenemos como parte de un gradiente y un factor
            grad_next = grad + (step * fact)

            # beta: sí
            beta = np.dot(grad_next, grad_next) / np.dot(grad, grad)

            # La dirección actualizada para el siguiente paso 
            direct = -grad_next + (beta * direct) 

            # ======= Actualizar =======
            grad = grad_next 
            print("Actualizamos el paso")
            k += 1
        
        return x 
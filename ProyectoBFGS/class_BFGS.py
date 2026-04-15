"""
#Algoritmo: July y Rapla 

ALGORITMO DE OPTIMIZACIÓN QUASI-NEWTON: BFGS (Broyden-Fletcher-Goldfarb-Shannon)
Referencia: algoritmo bfgs.jpg

1. INICIALIZACIÓN:
   - k = 0 (Contador de iteraciones)
   - x_k = x_0 (Punto inicial)
   - f_k = f(x_k) (Evaluación inicial de la función)
   - H_k = H_0 (Aproximación inicial de la inversa del Hesiano, usualmente la Identidad I)
   - g_k = grad(f(x_k)) (Gradiente inicial)
   - Parámetros de búsqueda: epsilon (tolerancia), phi, c1, c2

2. BUCLE PRINCIPAL (While STOP_CRIT() is False):
   - Criterios de parada: 
        * k < k_max (Máximo de iteraciones)
        * ||g_k|| < tau (Norma del gradiente menor a la tolerancia)

   a. Dirección de búsqueda:
      p_k = -H_k * g_k

   b. Búsqueda de línea (Line Search - While STEP_CRIT() is False):
      - Iniciar step length: alpha_k = c
      - Actualizar alpha_k: alpha_k = alpha_k * phi
      - Nuevo punto: x_{k+1} = x_k + alpha_k * p_k
      - Evaluar: f_{k+1} = f(x_{k+1})
      - Criterios posibles (STEP_CRIT): Wolf-1, Wolf-2, Wolf-strong, Goldstein o Armijo [1].

   c. Cálculo de vectores de actualización:
      s_k = x_{k+1} - x_k = alpha_k * p_k (Cambio en la posición) [1, 2]
      g_{k+1} = grad(f(x_{k+1})) (Nuevo gradiente)
      y_k = g_{k+1} - g_k (Cambio en el gradiente) [1, 2]

   d. Actualización de la inversa del Hesiano (Fórmula BFGS):
      rho_k = 1 / (s_k^T * y_k) [1, 3]
      H_{k+1} = (I - rho_k * s_k * y_k^T) * H_k * (I - rho_k * y_k * s_k^T) + rho_k * s_k * s_k^T [1, 4]

   e. Siguiente iteración:
      k = k + 1
      x_k = x_{k+1}
      g_k = g_{k+1}

3. RETORNO:
   - x* (Punto mínimo encontrado)
"""
"""

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
            #print("Encontro al ciclo")
            # Calculamos este factor antes para ahorrar operaciones
            # Esto es lo de Ap_x pero se está implementado de forma numérica en funcion.diof() 
            fact = self.funcion_obj.doif(direct) 

            # ======= Optimizar =======
            # de: (g_k^T * g_k) / (p_k^T * A * p_k)
            # Sacamos solo el denominador 
            #print("optimizar")
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
            #print("Actualizamos el paso")
            k += 1
        
        return x 
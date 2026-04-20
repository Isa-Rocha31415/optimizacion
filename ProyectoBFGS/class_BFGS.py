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


#from funcion import funcion
#import numpy as np

#class GC:
    
    #def __init__(self, funcion_obj: funcion, k_max=1000):
        
        #self.max_error = 1e-6 
        #self.funcion_obj = funcion_obj
        #self.k_max = k_max 

    
   
   ## Gradiente Conjugado
   
    #def solve(self, x_0:np.ndarray): 
   
"""
    Optimizador de gradiente conjugado.
    @args: 
        x_0:[array] = representa el punto inicial en la función objetivo 
        A : Una matriz simétrica positiva que sale de algún lugar
    @return: 
        x: ndarray = el punto mínimo encontrado 
   """
        #k = 0 
        #x = x_0.flatten().copy() 

        # g0 = Ax - b  
        #grad = self.funcion_obj.diff(x) 

        # p0 = -g0
        #direct:np.ndarray = -grad
        #rank  = x_0.size
        
        #while k < rank and np.linalg.norm(grad) > self.max_error:
            #print("Encontro al ciclo")
            # Calculamos este factor antes para ahorrar operaciones
            # Esto es lo de Ap_x pero se está implementado de forma numérica en funcion.diof() 
            #fact = self.funcion_obj.doif(direct) 

            # ======= Optimizar =======
            # de: (g_k^T * g_k) / (p_k^T * A * p_k)
            # Sacamos solo el denominador 
            #print("optimizar")
            #denom = np.dot(direct, fact) # <- Podría dar un división por 0 así que hay que reisar la consola
            
            # Calcular nuevo tamaño de paso            
            #step =  np.dot(grad, grad) / denom 

            # El siguiente punto para buscar será en la dirección de minimización encontrada
            #x = x + (step * direct) 

            # La dirección la tenemos como parte de un gradiente y un factor
            #grad_next = grad + (step * fact)

            # beta: sí
            #beta = np.dot(grad_next, grad_next) / np.dot(grad, grad)

            # La dirección actualizada para el siguiente paso 
            #direct = -grad_next + (beta * direct) 

            # ======= Actualizar =======
            #grad = grad_next 
            #print("Actualizamos el paso")
            #k += 1
        
        #return x 
    
##########################################################################
from funcion import funcion #De este modulo importamos la clase funcion para poder utilizarla(solo para no olvidar)
import numpy as np # sirve para hacer matemáticas con matrices

class BFGS:
    def __init__(self, funcion_obj, k_max=1000):
        # Guardamos la configuración inicial
        self.funcion_obj = funcion_obj # La función de las imágenes
        self.k_max = k_max             # Máximo de intentos
        self.tau = 1e-6                # Tolerancia (qué tan cerca del cero queremos llegar)

def solve(self, x_0):
        k = 0
        x_k = x_0.flatten().copy()  # 6 thetas iniciales
        I = np.eye(x_k.size)        # Matriz Identidad (1s en la diagonal)
        H_k = I.copy()              # H inicial es la Identidad [1]
        g_k = self.funcion_obj.diff(x_k) # Gradiente inicial [1]

while k < self.k_max and np.linalg.norm(g_k) > self.tau:
            # a) Dirección de búsqueda (pk = -Hk * gk)
            p_k = -np.dot(H_k, g_k) # Multiplicación de matriz H por vector g [1]

            # b) Búsqueda de línea (Line Search - Simplificado)
            alpha = 1.0 # Tamaño de paso inicial
            phi = 0.5   # Factor de reducción
            c1 = 1e-4   # Parámetro de Wolfe
            
            # Mientras no baje suficiente el error, hacemos mas chiquito el alpha
            while self.funcion_obj.eval(x_k + alpha * p_k) > \
                  self.funcion_obj.eval(x_k) + c1 * alpha * np.dot(g_k, p_k):
                alpha *= phi # Achicamos el paso
            
            # c. Cálculo de vectores de actualización
            s_k = alpha * p_k # Cambio en la posición [1, 2]
            x_next = x_k + s_k
            g_next = self.funcion_obj.diff(x_next) # Nuevo gradiente
            y_k = g_next - g_k # Cambio en el gradiente [1, 2]

# d) Esta es la ctualización de la inversa del Hesiano
            rho_k = 1.0 / np.dot(y_k, s_k) # Escalar de ajuste [1, 5]
            
            # Parte 1: (I - rho * s * y^T)
            A1 = I - rho_k * np.outer(s_k, y_k)
            # Parte 2: (I - rho * y * s^T)
            A2 = I - rho_k * np.outer(y_k, s_k)
            
            # Fórmula BFGS Completa: Hk+1 = A1 * Hk * A2 + rho * s * s^T
            H_k = np.dot(A1, np.dot(H_k, A2)) + rho_k * np.outer(s_k, s_k) [1, 3]

            # e) Aqui se hace la siguiente iteración
            k += 1
            x_k = x_next
            g_k = g_next
            
        return x_k # Retornamos los thetas ideales
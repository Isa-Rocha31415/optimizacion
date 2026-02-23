from funcion import funcion
import numpy as np
from typing import Union, List

class rosenbrock(funcion):
    # RAPLA  f(x_1, x_2, x_3, ..., x_n ) = Σ(1 - x_i)**2 + 100*(x_i+1 - x_i**2)**2

    def eval(self, args: np.ndarray) -> Union[int, float]:
        """
        Método que evalúa la función en un punto particular
        
        :param args: Punto n-dimensional para evaluar
        :type args: np.ndarray
        :return: Valor de la función en el punto dado 
        :rtype: Union[int, float]
        """
        result = 0.0

        for i in range(len(args) - 1): 
            x = args[i]
            y = args[i+1] 
            
            result += (1 - x)**2 + 100*(y - x**2)**2

        return result 
        
    def firstDimension(self, x: Union[int, float], x_n1: Union[int, float]) -> Union[int, float]: 
        result = -400*x*(x_n1 - x**2) - 2*(1 - x) 
        return result 

    def middleDimensions(self, args: np.ndarray) -> Union[int, float]: 
        # Implementación para dimensiones intermedias
        # Esta función debería calcular la derivada para índices intermedios
        pass 

    def lastDimension(self, x: Union[int, float], x_1n: Union[int, float]) -> Union[int, float]: 
        result = 200*(x - x_1n**2)
        return result
       
    def diff(self, args: np.ndarray) -> np.ndarray:
        """
        Función que evalúa el gradiente en un punto n-dimensional. 
        
        :param args: Punto en el que se evalúa la función.
        :type args: np.ndarray
        :return: Vector del gradiente
        :rtype: np.ndarray
        """
        # Para n-dimensional, necesitamos calcular todas las derivadas
        n = len(args)
        gradient = np.zeros(n)
        
        # Primera dimensión
        gradient[0] = -400*args[0]*(args[1] - args[0]**2) - 2*(1 - args[0])
        
        # Dimensiones intermedias
        for i in range(1, n-1):
            gradient[i] = 200*(args[i] - args[i-1]**2) - 400*args[i]*(args[i+1] - args[i]**2) - 2*(1 - args[i])
        
        # Última dimensión
        if n > 1:
            gradient[n-1] = 200*(args[n-1] - args[n-2]**2)
        
        return gradient

    def doiff(self, args: np.ndarray) -> np.ndarray: 
        """
        Función que obtiene la matriz Hessiana en un punto. 

        :param args: Punto para evaluar
        :type args: np.ndarray
        :return: Matriz Hessiana evaluada
        :rtype: np.ndarray
        """
        
        n = len(args)
        hessian = np.zeros((n, n))
        
        # Para Rosenbrock n-dimensional, la Hessiana es tridiagonal
        for i in range(n):
            if i > 0:
                # Términos fuera de la diagonal
                hessian[i, i-1] = -400 * args[i-1]
                hessian[i-1, i] = -400 * args[i-1]
            
            # Términos diagonales
            if i == 0:
                hessian[0, 0] = 1200*args[0]**2 - 400*args[1] + 2
            elif i == n-1:
                hessian[n-1, n-1] = 200
            else:
                hessian[i, i] = 1200*args[i]**2 - 400*args[i+1] + 202
        
        return hessian
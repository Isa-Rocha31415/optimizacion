from funcion import funcion
import numpy as np

class rosenbrock(funcion):
    # RAPLA  f(x, y) = (1 - x)**2 + 100*(y - x**2)**2

    def eval(self, args: np.array) -> int:
        """
        Método que evalúa la función en un punto particular
        
        :param args: Punto en el que se evalúa la función tal que (x, y)
        :type args: np.array
        :return: Valor de la función en el punto dado 
        :rtype: int
        """
        x = args[0] 
        y = args[1]

        result = (1 - x)**2 + 100*(y - x**2)**2
        return result 
        

    def diff(self, args : np.array) -> np.array:
        """
        Función que evalúa el gradiente en un punto. 
        
        :param args: Punto en el que se evalúa la función tal que (x, y)
        :type args: np.array
        :return: Vector del gradiente
        :rtype: np.array
        """

        x = args[0] 
        y = args[1]  

        # Derivadas parciales 
        dx_result = -2*(1 - x) - 400*x*(y - x**2)
        dy_result = 200*(y - x**2) 
        
        gradient =  np.array([dx_result, dy_result])  
        return gradient

    def doiff(self, args : np.array) -> np.array: 
        """
        Función que obtiene la matriz Hessiana en un punto. 

        :param args: Punto para evaluar tal que (x, y) 
        :type args: np.array
        :return: Matriz Hessiana evaluada
        :rtype: np.array
        """
        
        x = args[0]
        y = args[1] 

        # Derivadas parciales de x
        dxdx_result = 1200*(x**2) - 400*y + 2 
        dydx_result = -400*x 

        # Derivadas parciales de y
        dxdy_result = -400*x  
        dydy_result = 200 

        col_1 = np.array(dxdx_result, dydx_result) 
        col_2 = np.array(dxdy_result, dydy_result) 

        # Construcción de la matriz final
        hess_mat = np.array([
        [dxdx_result, dxdy_result],
        [dydx_result, dydy_result]
        ])
        return hess_mat
    
    pass
from funcion import funcion
import numpy as np

class rosenbrock(funcion):
    # RAPLA  f(x_1, x_2, x_3, ..., x_n ) = Σ(1 - x_i)**2 + 100*(x_i+1 - x_i**2)**2

    def eval(self, args: np.array) -> int | float:
        """
        Método que evalúa la función en un punto particular
        
        :param args: Punto n-dimensional para evaluar
        :type args: np.array
        :return: Valor de la función en el punto dado 
        :rtype: int
        """
        result = 0

        for i in range(len(args) - 1): 
            x = args[i]
            y = args[i+1] 
            
            result += (1 - x)**2 + 100*(y - x**2)**2

        return result 
        
    def firstDimension(x : int | float, x_n1: int | float) -> int | float: 
        result = -400*x*(x_n1 - x**2) - 2*(1 - x) 
        return result 

    def middleDimensions(args : np.array[int | float]) -> int | float: 
        pass 

    def lastDimension(x: int | float, x_1n: int | float) -> int | float: 
        result = 200*(x - x_1n**2)
        return result
       
    def diff(self, args : np.array) -> np.array:
        """
        Función que evalúa el gradiente en un punto n-dimensional. 
        
        :param args: Punto en el que se evalúa la función.
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
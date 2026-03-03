from funcion import funcion
import numpy as np
 
class griewangk(funcion):
    """
    Implementación de la función de Griewangk para 2 dimensiones.
    Minimo global en (0, 0) con valor 0.
    """
 
    def eval(self, args: np.array) -> float:
        """
        Método que evalúa la función Griewangk en un punto (x, y).
        :param args: Punto en el que se evalúa la función tal que (x, y)
        :type args: np.array
        :return: Valor de la función en el punto dado
        """
        x = args[0]
        y = args[1]
 
        term_1 = (x**2 + y**2) / 4000
        term_2 = np.cos(x) * np.cos(y / np.sqrt(2))
        result = term_1 - term_2 + 1
        return result
 
    def diff(self, args: np.array) -> np.array:
        """
        Función que evalúa el gradiente de Griewangk en un punto.
        :param args: Punto en el que se evalúa la función tal que (x, y)
        :type args: np.array
        :return: Vector del gradiente
        """
        x = args[0]
        y = args[1]
 
        # Derivadas parciales
        # df/dx = x/2000 + sin(x)*cos(y/sqrt(2))
        dx_result = (x / 2000) + np.sin(x) * np.cos(y / np.sqrt(2))
        # df/dy = y/2000 + (1/sqrt(2))*cos(x)*sin(y/sqrt(2))
        dy_result = (y / 2000) + (1 / np.sqrt(2)) * np.cos(x) * np.sin(y / np.sqrt(2))
 
        gradient = np.array([dx_result, dy_result])
        return gradient
 
    def doiff(self, args: np.array) -> np.array:
        """
        Función que obtiene la matriz Hessiana de Griewangk en un punto.
 
        :param args: Punto para evaluar tal que (x, y)
        :type args: np.array
        :return: Matriz Hessiana evaluada
        """
        x = args[0]
        y = args[1]
        sqrt2 = np.sqrt(2)
 
        # Segundas derivadas
        # d2f/dx2
        dxdx_result = (1 / 2000) + np.cos(x) * np.cos(y / sqrt2)
        # d2f/dxdy (Derivada mixta)
        dxdy_result = -(1 / sqrt2) * np.sin(x) * np.sin(y / sqrt2)
        # d2f/dydx (Igual a dxdy por teorema de Clairaut)
        dydx_result = dxdy_result
        # d2f/dy2
        dydy_result = (1 / 2000) + (1 / 2) * np.cos(x) * np.cos(y / sqrt2)
 
        # Construcción de la matriz Hessiana
        hess_mat = np.array([
            [dxdx_result, dxdy_result],
            [dydx_result, dydy_result]
        ])
        return hess_mat
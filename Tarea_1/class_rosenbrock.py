from funcion import funcion
class rosenbrock(funcion):
    # RAPLA  f(x, y) = (1 - x)**2 + 100*(y - x**2)**2

    def eval(self, x):
        #calcular el valor de la función en un punto x. Ejemplo: return x[0]**2 + x[1]**2
        pass

    def diff(self, x):
        #Aquí debes calcular el gradiente (derivadas parciales).
        #Debes retornar un array de NumPy con las derivadas respecto a cada variable.
        pass

    def doiff(self, x):
        #Aquí debes calcular la segunda derivada (Hessiana).
        #Retorna una matriz con las segundas derivadas.
        pass
    
    pass

# Clase donde va la funcion
# Cris Gradiente
# Tadeo Tadeo
# Isa  Eval
from funcion import funcion
import numpy as np

# Vamos a utilizar condiciones de barrera- Toroide

class funcion(funcion):


    def eval(args):
        """
        Isa:
        """
        #nada
        pass
    
    def diff(self, x, h=1e-7): #h es un valor estandar para gradiente

        x = np.array(x, dtype=float)
        n = len(x)
        g = np.zeros(n)

        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = h

            f_adelante = self.eval(x + e_i)
            f_atras = self.eval(x - e_i)
            g[i] = (f_adelante - f_atras) / (2 * h)

        return g
        

    def doif():
        """
        Tadeo
        """
        #nada
        pass 

    pass
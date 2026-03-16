"""
FUNCIÓN DE COSTO PARA SUAVIZADO DE IMÁGENES (Regularización de Tikhonov)
-----------------------------------------------------------------------
E(X) = Σ [ (X_i,j - I_i,j)^2 + λ * ( (X_i,j - X_i+1,j)^2 + 
                                     (X_i,j - X_i-1,j)^2 + 
                                     (X_i,j - X_i,j+1)^2 + 
                                     (X_i,j - X_i,j-1)^2 ) ]

Donde:
- (X_i,j - I_i,j)^2 : Término de FIDELIDAD (mantiene la imagen cerca de la original).
- λ (lambda)        : PARÁMETRO DE REGULARIZACIÓN (controla la fuerza del suavizado).
- Términos vecinos  : PENALIZACIÓN POR DIFERENCIAS (suaviza el ruido).

GRADIENTE LOCAL (para optimización):
∂E/∂X_i,j = 2(X_i,j - I_i,j) + 2*λ * [ 4*X_i,j - (X_i+1,j + X_i-1,j + X_i,j+1 + X_i,j-1) ]

# Clase donde va la funcion
# Cris Gradiente
# Tadeo Tadeo
# Isa  Eval

"""


from funcion import funcion
import numpy as np

# Vamos a utilizar condiciones de barrera- Toroide

class funcion(funcion):


    def eval(args):
        """
        Isa:
        """
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
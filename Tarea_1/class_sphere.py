import numpy as np
from funcion import funcion

class sphere(funcion):
    
    def eval(self, x):
        return np.sum(np.array(x)**2) 

    def diff(self, x):
        return 2 * np.array(x)

    def doiff(self, x):
        x = np.array(x)
        n = len(x)
        return 2 * np.identity(n)
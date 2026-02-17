from funcion import funcion
import numpy as np
class cigar(funcion):
    #  JULY f(x) = sum(xi^2)

    def eval(self, x):
        return x[0]**2 + 10**6 * np.sum(x[1:]**2)
    
    def diff(self, x):
        grad = 2 * x * (10**6)
        grad[0] = 2 * x[0]
        return grad
    
    def doiff(self, x):
        h = np.eye(len(x)) * 2 * (10**6)
        h[0, 0] = 2
        return h
    
    pass
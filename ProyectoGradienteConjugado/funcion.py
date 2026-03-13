import numpy as np

# Clase padre solo e sun modelo
class funcion: 
    def eval(args):
        #nada
        pass
    
    def diff():
        #nada
        pass 

    def doif():
        #nada
        pass 

    def plot(self, lim, canvas):
        x_vals = np.linspace(lim[0], lim[1], 300)
        y_vals = np.linspace(lim[2], lim[3], 300)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.eval(np.array([X[i, j], Y[i, j]]))
        #curvas
        canvas.contour(X, Y, Z, levels=50, cmap='viridis')
        canvas.set_xlabel('x1')
        canvas.set_ylabel('x2')

        pass
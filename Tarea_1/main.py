import matplotlib.pyplot as plt
import numpy as np
from class_sphere import sphere
from class_cigar import cigar
from class_rosenbrock import rosenbrock
from class_dg import dg

if __name__ == "__main__":

    #unimos todo:
    fig, plot = plt.subplots(figsize=(10,8))

    # por cada funcion
    def parametros (func,alpha,k_max, lim_xy, lim_1, lim_2):
        desenso_gradiente =dg(func, alpha =alpha,k_max =k_max)
        punto_inicial = np.random.uniform(lim_1, lim_2, size=2)
        desenso_gradiente.solve(x0 =punto_inicial)

        #func.plot(lim=[-50,50,-100,100],canvas=plot)
        func.plot(lim=lim_xy,canvas=plot)


        desenso_gradiente.plot2d(canvas=plot)

        plt.title(f"Optimizacion de {type(func).__name__}")
        plt.show()

    #parametros(rosenbrock(), 0.0001, 3000,[-20,10,-20,10], -5,5)
    #parametros(cigar(), 0.000001, 1000,[-20,10,-20,10],-10,10)
    parametros(sphere(), 0.1, 1000,[-200,200,-200,200],-100,100)

    pass
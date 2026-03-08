import matplotlib.pyplot as plt
import numpy as np
from class_sphere import sphere
from class_cigar import cigar
from class_rosenbrock import rosenbrock
from class_griewangk import griewangk
from class_dg import dg
from condicion_goldstein import c_goldstein
from condicion_curvatura import curvatura
from condicion_fuertewolfe import c_fuertewolfe
from condicion_armijo import c_armijo
from condicion_wolfe import c_wolfe

if __name__ == "__main__":

    #unimos todo:
    fig, plot = plt.subplots(figsize=(10,8))

    # por cada funcion
    def parametros_armijo (func, alpha, k_max, lim_xy, lim_1, lim_2):
        desenso_gradiente = dg(func, alpha=alpha, k_max=k_max, tolerancia=1e-5)

        desenso_gradiente.setCondition(c_armijo)
        punto_inicial = np.random.uniform(lim_1, lim_2, size=2)
        desenso_gradiente.solve(x0=punto_inicial)
        func.plot(lim=lim_xy, canvas=plot)
        desenso_gradiente.plot2d(canvas=plot)

        plt.title(f"Optimización de {type(func).__name__} Armijo")
        plt.show()

    def parametros_wolfe(func, alpha, k_max, lim_xy, lim_1, lim_2):
        desenso_gradiente = dg(func, alpha=alpha, k_max=k_max, tolerancia=1e-5)

        desenso_gradiente.setCondition(c_wolfe)
        punto_inicial = np.random.uniform(lim_1, lim_2, size=2)
        desenso_gradiente.solve(x0=punto_inicial)
        func.plot(lim=lim_xy, canvas=plot)
        desenso_gradiente.plot2d(canvas=plot)

        plt.title(f"Optimización de {type(func).__name__} Wolfe")
        plt.show()

    def parametros_wolfefuerte(func, alpha, k_max, lim_xy, lim_1, lim_2):
        desenso_gradiente = dg(func, alpha=alpha, k_max=k_max, tolerancia=1e-5)

        desenso_gradiente.setCondition(c_fuertewolfe)
        punto_inicial = np.random.uniform(lim_1, lim_2, size=2)
        desenso_gradiente.solve(x0=punto_inicial)
        func.plot(lim=lim_xy, canvas=plot)
        desenso_gradiente.plot2d(canvas=plot)

        plt.title(f"Optimización de {type(func).__name__} Fuuerte Wolfe")
        plt.show()

    def parametros_c_goldstein(func, alpha, k_max, lim_xy, lim_1, lim_2):
        desenso_gradiente = dg(func, alpha=alpha, k_max=k_max, tolerancia=1e-5)

        desenso_gradiente.setCondition(c_goldstein)
        punto_inicial = np.random.uniform(lim_1, lim_2, size=2)
        desenso_gradiente.solve(x0=punto_inicial)
        func.plot(lim=lim_xy, canvas=plot)
        desenso_gradiente.plot2d(canvas=plot)

        plt.title(f"Optimización de {type(func).__name__} condicion goldstein")
        plt.show()

    def parametros_curvatura(func, alpha, k_max, lim_xy, lim_1, lim_2):
        desenso_gradiente = dg(func, alpha=alpha, k_max=k_max, tolerancia=1e-5)

        desenso_gradiente.setCondition(curvatura)
        punto_inicial = np.random.uniform(lim_1, lim_2, size=2)
        desenso_gradiente.solve(x0=punto_inicial)
        func.plot(lim=lim_xy, canvas=plot)
        desenso_gradiente.plot2d(canvas=plot)

        plt.title(f"Optimización de {type(func).__name__} condicion goldstein")
        plt.show()



    #parametros(rosenbrock(), 0.0001, 3000,[-20,10,-20,10], -5,5)
    #parametros(cigar(), 0.000001, 1000,[-20,10,-20,10],-10,10)
    parametros_curvatura(griewangk(), 0.0001, 1000,[-8,8,-8,8],-8,8)


    pass
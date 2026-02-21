import matplotlib.pyplot as plt
import numpy as np
from class_sphere import sphere
from class_cigar import cigar
from class_rosenbrock import rosenbrock
from class_dg import dg

# Definir lista de dimensiones R y rango de alpha
# Lista R (Dimensiones): [2, 5, 10, 15, 20]
dimensiones_R = [2, 5, 10, 15, 20] 

# Lista de 10 alpha randoms entre (0.5 y 2)
alpha_valores = np.random.uniform(0.5, 2, size=10) 

# Lista k-max: [1000, 500, 1500, 2000]
k_max_valores = [1000, 500, 1500, 2000]

# Lista de punto_inicial: tamaño 10 puntos iniciales
# Nota: Se generan 10 puntos (ajustado según la última nota) con la dimensión actual
puntos_iniciales_lista = [np.random.uniform(-100, 100, size=dimensiones_R[0]) for _ in range(10)]

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
    
    # Ejemplo usando los nuevos valores de las listas
    parametros(sphere(), alpha_valores[0], k_max_valores[0], [-200,200,-200,200], -100, 100)
    
    # hacer un grsfico en 3d para especialmente la funcion cigar

    pass


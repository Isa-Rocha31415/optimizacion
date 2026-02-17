import matplotlib.pyplot as plt
import numpy as np

from class_sphere import sphere
from class_cigar import cigar
from class_rosenbrock import rosenbrock
from class_dg import dg

if __name__ == "__main__":

    #unimos todo:
    fig, plot = plt.subplots(figsize=(10,8))

    #func = sphere
    #func = cigar
    func = rosenbrock

    desenso_gradiente =dg(func,paso=0.0004)
    punto_inicial = [0,0]
    desenso_gradiente.solve(x_o =punto_inicial)

    func.plot(lim=[-5,5], canvas =ax)
    desenso_gradiente.plot2d(canvas=ax)

    plt.title(f"Optimizacion de {type(func).__name__}")
    plt.show()

    pass
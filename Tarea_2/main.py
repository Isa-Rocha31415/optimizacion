import pandas as pd
import numpy as np
from class_sphere import sphere
from class_cigar import cigar
from class_rosenbrock import rosenbrock
from class_dg import dg


dimensiones_R = [2, 5, 10, 15, 20]

alpha_valores = np.random.uniform(0.001, 0.01, size=10)

k_max = 25


if __name__ == "__main__":

    funciones = [sphere, cigar, rosenbrock]
    resultados = []

    for funcion_clase in funciones:

        for R in dimensiones_R:

            mejor_alpha = None
            mejor_promedio = float("inf")

            for alpha in alpha_valores:

                xi_lista = []

                for _ in range(30):   

                    if funcion_clase == sphere:
                        x0 = np.random.uniform(-600, 300, size=R)

                    elif funcion_clase == cigar:
                        x0 = np.random.uniform(-20, 10, size=R)

                    elif funcion_clase == rosenbrock:
                        x0 = np.random.uniform(-20, 10, size=R)

                    func = funcion_clase()
                    optimizador = dg(func, alpha=alpha, k_max=k_max)

                    x_optimo = optimizador.solve(x0=x0)

                    xi = np.linalg.norm(x0 - x_optimo) ** 2
                    xi_lista.append(xi)

                promedio_xi = np.mean(xi_lista)

                resultados.append([
                    funcion_clase.__name__,
                    R,
                    promedio_xi,
                    alpha
                ])

    tabla = pd.DataFrame(
        resultados,
        columns=["Funcion", "Dimension", "Promedio_Xi", "Alpha_mejor"]
    )

    tabla = tabla.groupby("Funcion").sample(n=10, random_state=42).reset_index(drop=True)

    print(tabla)
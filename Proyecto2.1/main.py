import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


dimensiones_R = [2, 5, 10, 15, 20]
k_max = [300, 500, 1000, 1500, 2000]
condiciones = [c_goldstein,curvatura] #falta agregar algunas
alpha = 0.1
if __name__ == "__main__":
    funciones = [sphere, cigar]
    resultados = []

    for funcion_clase in funciones:
        for R in dimensiones_R:
            # Iteramos sobre k_max iteraciones
            for C in condiciones:
                #iteramos sobre las condiciones
                for k in k_max:
                    
                    lista_errores = []
                    lista_k = []
                    
                    for _ in range(30):   
                        
                        if funcion_clase == sphere:
                            x0 = np.random.uniform(-600, 300, size=R)
                        elif funcion_clase == cigar:
                            x0 = np.random.uniform(-20, 10, size=R)
                        elif funcion_clase == rosenbrock:
                            x0 = np.random.uniform(-2, 2, size=R)
                        elif funcion_clase == griewangk:
                            x0 = np.random.uniform(-8, 8, size=R)

                        func = funcion_clase()
                        optimizador = dg(func, alpha=alpha, k_max=k,tolerancia=1e-5)
                        optimizador.setCondition(C)

                        trayectoria, k_final = optimizador.solve(x0=x0)
                        punto_final = trayectoria[-1]
                        
                        # Calculamos el error (distancia al origen al cuadrado).
                        error_actual = np.linalg.norm(punto_final) ** 2
                        
                lista_errores.append(error_actual)
                lista_k.append(k_final)
                performance =np.average(lista_k)
                accuracy = np.average(lista_errores)



                #guardamos el mejor resumen 
                resultados.append([
                    funcion_clase.__name__,
                    R,
                    C.__name__,
                    performance,
                    accuracy
                ])

    tabla = pd.DataFrame(
        resultados,
        columns=["Funcion", "Dimension", "Condicion", "Performance", "Acurracy"]
    )

    tabla = tabla.sort_values(by=["Funcion", "Dimension","Condicion"]).reset_index(drop=True)

    print(tabla)


    pass
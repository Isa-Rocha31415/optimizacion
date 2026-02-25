import pandas as pd
import numpy as np
from class_sphere import sphere
from class_cigar import cigar
from class_rosenbrock import rosenbrock
from class_dg import dg

dimensiones_R = [2, 5, 10, 15, 20]
k_max = [300, 500, 1000, 1500, 2000]


np.random.seed(42)

alpha_valores = np.random.uniform(0.000000000001, 0.000001, size=10)

if __name__ == "__main__":
    funciones = [sphere, cigar, rosenbrock]
    resultados = []

    for funcion_clase in funciones:
        for R in dimensiones_R:
            
            mejor_error_global = float('inf')
            mejor_alpha_global = None
            mejor_k_promedio = 0
            mejor_k_max_config = 0
            
            # Iteramos sobre k_max
            for k in k_max:
                for alpha in alpha_valores:
                    
                    lista_errores = []
                    lista_k = []
                    
                    for _ in range(30):   
                        
                        if funcion_clase == sphere:
                            x0 = np.random.uniform(-600, 300, size=R)
                        elif funcion_clase == cigar:
                            x0 = np.random.uniform(-20, 10, size=R)
                        elif funcion_clase == rosenbrock:
                            x0 = np.random.uniform(-20, 10, size=R)

                        func = funcion_clase()
                        optimizador = dg(func, alpha=alpha, k_max=k)

                        trayectoria, k_final = optimizador.solve(x0=x0)
                        punto_final = trayectoria[-1]
                        
                        # Calculamos el error (distancia al origen al cuadrado).
                        error_actual = np.linalg.norm(punto_final) ** 2
                        
                        lista_errores.append(error_actual)
                        lista_k.append(k_final)

                    promedio_error = np.mean(lista_errores)
                    promedio_k = np.mean(lista_k)

                    # Buscamos MINIMIZAR el error (promedio_xi)
                    if promedio_error < mejor_error_global:
                        mejor_error_global = promedio_error
                        mejor_alpha_global = alpha
                        mejor_k_promedio = promedio_k
                        mejor_k_max_config = k


            #guardamos el mejor resumen 
            resultados.append([
                funcion_clase.__name__,
                R,
                mejor_k_promedio,
                mejor_error_global, #Error minimo logrado)
                mejor_alpha_global
            ])

    tabla = pd.DataFrame(
        resultados,
        columns=["Funcion", "Dimension", "Performance(promedio k iteraciones)", "Promedio_Xi", "Alpha_optimo"]
    )

    tabla = tabla.sort_values(by=["Funcion", "Dimension"]).reset_index(drop=True)

    print(tabla)
    print("\nValores de Alpha probados:", alpha_valores)
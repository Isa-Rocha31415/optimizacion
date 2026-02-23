import pandas as pd
import numpy as np
# Asumo que estas clases existen en tus archivos importados
from class_sphere import sphere
from class_cigar import cigar
from class_rosenbrock import rosenbrock
from class_dg import dg

# Configuración de parámetros
dimensiones_R = [2, 5, 10, 15, 20]
k_max = [300, 500, 1000, 1500, 2000]

# Generamos los alphas una vez para usarlos consistentemente
# Es mejor usar una semilla para que los resultados sean reproducibles
np.random.seed(42)
alpha_valores = np.random.uniform(0.000000000001, 0.000001, size=10)

if __name__ == "__main__":
    funciones = [sphere, cigar, rosenbrock]
    resultados = []

    for funcion_clase in funciones:
        for R in dimensiones_R:
            
            # Variables para guardar el MEJOR resultado de todas las combinaciones de alpha y k
            # para esta Funcion y Dimension específica.
            mejor_error_global = float('inf') # Inicializamos en infinito para poder minimizar
            mejor_alpha_global = None
            mejor_k_promedio = 0
            mejor_k_max_config = 0
            
            # Iteramos sobre las configuraciones de k_max
            for k in k_max:
                for alpha in alpha_valores:
                    
                    lista_errores = []
                    lista_k = []
                    
                    # Monte Carlo: 30 repeticiones
                    for _ in range(30):   
                        
                        # Definir x0 según la función
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
                        
                        # CORRECCIÓN DE METRICÁ:
                        # Calculamos el error (distancia al origen al cuadrado).
                        # Asumimos que el mínimo está en el origen (0,0,...).
                        # Para Rosenbrock el mínimo está en (1,1,...), pero para simplificar 
                        # y comparar rendimiento general usamos la norma del punto final.
                        # Cuanto más cerca de 0, mejor.
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

            # NOTA: El append va FUERA de los bucles de alpha y k, 
            # pero DENTRO del bucle de dimensiones.
            # Guardamos el mejor resumen para esta combinación Función-Dimensión
            resultados.append([
                funcion_clase.__name__,
                R,
                mejor_k_promedio,
                mejor_error_global, # Este es el "Promedio_Xi" (Error mínimo logrado)
                mejor_alpha_global
            ])

    # Creación de la tabla final
    tabla = pd.DataFrame(
        resultados,
        columns=["Funcion", "Dimension", "Performance(promedio k iteraciones)", "Promedio_Xi", "Alpha_optimo"]
    )

    # No necesitamos samplear. Queremos ver todas las combinaciones (15 filas).
    # Ordenamos para mejor visualización
    tabla = tabla.sort_values(by=["Funcion", "Dimension"]).reset_index(drop=True)

    print(tabla)
    print("\nValores de Alpha probados:", alpha_valores)
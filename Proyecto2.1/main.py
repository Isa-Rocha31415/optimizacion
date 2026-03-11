import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Asumimos que estas clases y funciones existen en tu entorno
from class_sphere import sphere
from class_cigar import cigar
from class_rosenbrock import rosenbrock
from class_griewangk import griewangk
from class_dg import dg
from condicion_goldstein import c_goldstein
from condicion_curvatura import curvatura
from condicion_armijo import c_armijo
from condicion_wolfe import c_wolfe
from condicion_fuertewolfe import c_fuertewolfe

dimensiones_R = [2, 5, 10, 15, 20]
# Nota: Como vamos a promediar los resultados de k_max, estos se incluyen en el 'pool' de datos
k_max_values = [300, 500, 1000, 1500, 2000] 
condiciones = [c_goldstein, curvatura, c_armijo, c_wolfe, c_fuertewolfe]
alpha = 0.1

if __name__ == "__main__":
    funciones = [sphere, cigar, rosenbrock, griewangk]
    
    # CAMBIO IMPORTANTE: Usaremos una lista plana para guardar CADA ejecución individual
    datos_crudos = []

    print("Iniciando simulaciones... (Esto puede tardar un poco)")

    for funcion_clase in funciones:
        for R in dimensiones_R:
            for C in condiciones:
                for k in k_max_values:
                    
                    # Hacemos las 3 repeticiones
                    for _ in range(30):   
                        
                        # Definición de x0 según la función
                        if funcion_clase == sphere:
                            x0 = np.random.uniform(-600, 300, size=R)
                        elif funcion_clase == cigar:
                            x0 = np.random.uniform(-20, 10, size=R)
                        elif funcion_clase == rosenbrock:
                            x0 = np.random.uniform(-2, 2, size=R)
                        elif funcion_clase == griewangk:
                            x0 = np.random.uniform(-8, 8, size=R)

                        func = funcion_clase()
                        # Instanciamos el optimizador
                        optimizador = dg(func, alpha=alpha, k_max=k, tolerancia=1e-5)
                        optimizador.setCondition(C)

                        # Resolver
                        trayectoria, k_final = optimizador.solve(x0=x0)
                        punto_final = trayectoria[-1]
                        
                        # Calculamos métricas
                        error_actual = np.linalg.norm(punto_final) ** 2
                        
                        # Guardamos el resultado crudo de esta ejecución específica
                        # Incluimos 'k_max' para poder filtrar o verificar, aunque luego lo promediaremos
                        datos_crudos.append({
                            "Funcion": funcion_clase.__name__,
                            "Dimension": R,
                            "Condicion": C.__name__,
                            "k_max_config": k,   # Parámetro de configuración
                            "Iteraciones": k_final, # Performance real
                            "Error": error_actual   # Accuracy
                        })

    print("Simulaciones terminadas. Procesando datos...")

    # 1. Creamos el DataFrame con todos los datos crudos
    df_crudo = pd.DataFrame(datos_crudos)

    # 2. Agrupamos y Promediamos (Colapsar k_max)
    # Agrupamos por Funcion, Dimension y Condicion.
    # Esto promedia automáticamente las 3 repeticiones Y los 5 valores de k_max (total 15 muestras por fila)
    tabla_agrupada = df_crudo.groupby(["Funcion", "Dimension", "Condicion"]).agg(
        Performance=("Iteraciones", "mean"),
        Accuracy=("Error", "mean")
    ).reset_index()

    # Ordenamos la tabla final
    tabla_agrupada = tabla_agrupada.sort_values(by=["Funcion", "Dimension", "Condicion"]).reset_index(drop=True)

    print("\n--- Tabula Resumen (Promediada) ---")
    print(tabla_agrupada)

    # 3. Exportar a LaTeX
    # Usamos formateo científico para Accuracy porque suelen ser números muy pequeños
    latex_tabla = tabla_agrupada.to_latex(
        index=False, 
        caption="Resultados promedio de Performance y Accuracy (Promediando $k_{max}$ y repeticiones)",
        label="tab:resultados_optimizacion_agrupados",
        column_format="|l|c|c|r|r|",
        position="ht",
        float_format="%.2e" # Notación científica para los números flotantes
    )

    print("\n--- Código LaTeX ---")
    print(latex_tabla)
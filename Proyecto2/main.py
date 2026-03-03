import matplotlib.pyplot as plt
import numpy as np
from class_sphere import sphere
from class_cigar import cigar
from class_rosenbrock import rosenbrock
from class_griewangk import griewangk
from class_dg import dg
from condicion_goldstein import c_goldstein
if __name__ == "__main__":

    #unimos todo:
    fig, plot = plt.subplots(figsize=(10,8))

    # por cada funcion
    def parametros (func, alpha, k_max, lim_xy, lim_1, lim_2):
        # 1. Instanciar el optimizador
        # Nota: Asegúrate que tu clase dg reciba 'tolerancia' si así lo definiste en el constructor
        desenso_gradiente = dg(func, alpha=alpha, k_max=k_max, tolerancia=1e-5)
        
        # 2. CONFIGURAR la condición ANTES de resolver
        # Aquí le pasas la función que importaste
        desenso_gradiente.setCondition(c_goldstein)
        
        # 3. Punto inicial aleatorio
        punto_inicial = np.random.uniform(lim_1, lim_2, size=2)
        
        # 4. EJECUTAR la optimización
        desenso_gradiente.solve(x0=punto_inicial)
        
        # 5. GRAFICAR los resultados
        func.plot(lim=lim_xy, canvas=plot)
        desenso_gradiente.plot2d(canvas=plot)

        plt.title(f"Optimización de {type(func).__name__} con Goldstein")
        plt.show()

    #parametros(rosenbrock(), 0.0001, 3000,[-20,10,-20,10], -5,5)
    #parametros(cigar(), 0.000001, 1000,[-20,10,-20,10],-10,10)
    parametros(cigar(), 0.001, 1000,[-100,100,-100,100],-100,100)

    pass
import pandas as pd
import numpy as np
import math

class dg:
    def __init__(self, funcion_obj, alpha=0.01, k_max=1000):
        
        self.funcion_obj = funcion_obj
        self.alpha = alpha
        self.k_max = k_max
        self.trayectoria = None
        
    def solve(self, x0):
        """
        Funcion desenso de gradiente
        1) k =0 , k_max
        2) X_a = X_0
        3) a_c  f(x_a)
        4) a_c = c
        5) DO
        6) P_x = -(Nabla)f(x_a)
        7) x_a1 = x_a + a_a Px
        8) f_k+1 = f(X_k+1)
        9) k =k+1
        10) WHILE(f_k+1 < f_a AND k <= K_max)
        """
        # 1) y 2) inicializacion
        x_a = np.array(x0, dtype=float)
        k =0
        resultados = [x_a]

        # 3) evaluamos f(x_a)
        f_a = self.funcion_obj.eval(x_a)

        # 5) bucle 
        while k < self.k_max:
            #6) P_x = -Gradiente
            p_x = -self.funcion_obj.diff(x_a)

            #7) x_a1 = x_a + alpha * P_x
            x_nueva = x_a + self.alpha * p_x

            #8) f_k+1 = f(x_nuevo)
            f_nueva = self.funcion_obj.eval(x_nueva)

            # 10) condicionales del ciclo
            # Si la nueva fun NO es menor que la anterior
            if not (f_nueva < f_a):
                break

            # actualizamos
            x_a = x_nueva
            f_a = f_nueva
            k += 1
            resultados.append(x_a)
            print(f"Paso {k}: {x_a}")
            
        self.trayectoria = np.array(resultados)
        print(f"Paso {k}: {x_a}")

    def plot2d(self, canvas):
        """
        ISA: Tomas los puntos guardados en self.history y los dibujas
        sobre el gráfico que preparó Cris usando canvas.plot().
        """
        puntos = np.array(self.trayectoria)
        canvas.plot(puntos[:, 0], puntos[:, 1], "r-", label ="Trayectoria Desenso de Gradiente")
        canvas.plot(puntos[0,0], puntos[0,1], "go", label="Inicio")
        canvas.plot(puntos[-1,0], puntos[-1,1], "bx", label = "Final")
        canvas.legend() 
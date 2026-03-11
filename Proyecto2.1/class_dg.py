import pandas as pd
import numpy as np
import funcion
class dg:

    def __init__(self, funcion_obj, tolerancia, reductor=0.5, alpha=0.01, k_max=1000):
        
        self.tolerancia = tolerancia
        self.alpha_inicial = 1 
        self.funcion_obj = funcion_obj # Rosenbrock, Cigar, sphere, griewangk
        self.alpha = alpha 
        self.k_max = k_max 
        self.rho = reductor
        self.trayectoria = []
 
        self.tolerancia = tolerancia # delta δ 
        self.condicion = None # función: Wolfe, Armijo, Goldstein... 


    def setCondition(self, condition): 
        self.condicion = condition 

    def solve(self, x0):
        #print('Entró a optimizar')
        """
        # Algorimto de Newton con Backtracking 
        1) k , k_max, α, δ, ρ
        2) x_a = x_0
        3) f_k = f(x_a)
        4) g_k = ∇f(x_k) 
        5) WHILE (||g_k|| > δ AND k < k_max) DO: 
        6)    α_k = c
        7)    H_k = Hf(x_k) 
        8)    P_k = - H_k^(-1) * g_k
        9)    x_{k+1} = x_k + α_k * P_k 
        10)   f_{k+1} = f_(x_{k+1})
        11)   WHILE (CONDICIÓN) DO: 
        12)        α_k =  ρ * α_k
        13)        x_{k+1} = x_k + α_k * P_k 
        14)        f_{k+1} = f_(x_{k+1})
        15)   ENDWHILE
        16)   g_{k+1} = ∇f(x_{k+1})
        17)   k++ 
        18)   ENDWHILE 
        19)   RETURN x_k   
        """

        # 1) La mayoría de las constantes están definidas en el constructor
        k = 0 # Sin iteraciones

        # 2)  Punto inicial
        x_k = np.array(x0, float) 
        self.trayectoria.append(np.copy(x_k))

        # 4) Evaluar el gradiente
        g_k = self.funcion_obj.diff(x_k)

        # 5) Optimizador: 
        while np.linalg.norm(g_k) > self.tolerancia and k < self.k_max: 
            #print('Entró al segundo ciclo')
            # 6) primer valor de alpha
            alpha_k  = self.alpha_inicial 

            # 7) Obtener la Hessiana en el punto inicial 
            H_f = self.funcion_obj.doiff(x_k)

            # 8) Resulta que es más rápido tratar el paso 8 como un sistema de la forma Ax = b que
            # sale al despejar -g_x y queda como: H_x * P_k = -g_x y sale más rapido con np.linalg.solve() 
            P_k = np.linalg.solve(H_f, -g_k) # P_k es la dirección en la que se mueve ahora que está dada por la curvatura

            # 9) Dar un paso grande con confianza de que es por ahí 
            x_k1 = x_k + alpha_k * P_k



            # 10) Evaluar la función en ese punto, o sea, medir la altura de la «montaña»
            f_k1 = self.funcion_obj.eval(x_k1)
            f_k = self.funcion_obj.eval(x_k)
            g_k1 = self.funcion_obj.diff(x_k1)
            # Neceistamos «donde estaba» y «a donde llegó para la condición»
            i = 0
            # 11) Encontrar el mejor tamaño para alpha 
            while not self.condicion(f_k, f_k1, g_k, g_k1, P_k, alpha_k) and i < 50: 
                # El si la condición se cumple, quiere decir que el paso no fue bueno, que no llegó tan abajo como debería
                # entonces rectifica. 

                # 12) Va a dar un paso la mitad de grande porque rho = 1/2
                alpha_k *= self.rho 
                
                # 13) Se regresa al paso anterior y da un paso más cortito en la dirección de P 
                x_k1 = x_k + alpha_k * P_k

                # 14) Volvemos a medir si estamos suficientemente más abajo.  
                f_k1 = self.funcion_obj.eval(x_k1)
                g_k1 = self.funcion_obj.diff(x_k1)
                # Repetir hasta que el paso sea bueno, que nos deje lo suficintenmente más abajo. 
                i+=1

            # 15) Calcular la nuva dirección para bajar a partir de estemos luego del pasito corto. 
            x_k = x_k1 # y actualizar 
            g_k = self.funcion_obj.diff(x_k1) 
            
            # 16) Contamos una iteración
            self.trayectoria.append(np.copy(x_k))
            k += 1
            #print(f"Paso {k}: {x_k}")
            #print(f'Hizo {i} vueltas del segundo while')

        self.trayectoria = np.array(self.trayectoria)
        return self.trayectoria, k


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
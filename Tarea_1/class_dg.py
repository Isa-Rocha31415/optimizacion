import pandas as pd
import numpy as np
import math

class dg:
    def __init__(self, funcion, paso):
        """
         Guardas la función que vas a optimizar y el paso (learning rate).
        Crea una lista vacía 'self.history' para guardar la trayectoria.
        """
        self.f = funcion
        self.paso = paso
        self.trayectoria =[]
        

    def solve(self, x_o, iteraciones=2000,tolerancia=0.0001):
        """
        Funcion desenso de gradiente
        """
        x = list(x_o)
        self.trayectoria = [list(x)]
        distancia =0
        for i in range(iteraciones):
            # gradiente
            gradiente = self.f.diff(x)

            #calcular el nuevo punto
            x_nuevo = []
            for j in range(len(x)):
                nuevo_valor =x[j] -self.paso * gradiente[j]
                x_nuevo.append(nuevo_valor)

                #calcular la distancia
                suma_cuadrados =0
            for k in range(len(x)):
                diferencia = x_nuevo[k] - x[k]
                suma_cuadrados +=diferencia **2

            distancia = math.sqrt(suma_cuadrados)
            self.trayectoria.append(list(x_nuevo))
            if distancia < tolerancia:
                print(f"Converge en {len(self.trayectoria)} pasos")
                break
            #actualizamos
            x = x_nuevo
            
        return x

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
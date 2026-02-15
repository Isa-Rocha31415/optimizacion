class DG:
    def __init__(self, funcion, c=1):
        """
        ISA: Guardas la función que vas a optimizar y el paso (learning rate).
        Crea una lista vacía 'self.history' para guardar la trayectoria.
        """
        pass

    def solve(self, x_o):
        """
        ISA: El algoritmo principal.
        Mientras no se alcance el máximo de iteraciones:
        1. Calculas el nuevo x usando la fórmula: $x_{new} = x - c \cdot \nabla f(x)$
        2. Guardas el punto en self.history.
        """
        pass

    def plot_2d(self, canvas):
        """
        ISA: Tomas los puntos guardados en self.history y los dibujas
        sobre el gráfico que preparó Cris usando canvas.plot().
        """
        pass
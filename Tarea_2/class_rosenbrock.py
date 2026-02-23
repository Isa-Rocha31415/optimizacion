from funcion import funcion
import numpy as np


class rosenbrock(funcion):

    # f(x) = sum_{i=1}^{n-1} (1 - x_i)^2 + 100 (x_{i+1} - x_i^2)^2

    def eval(self, args: np.array) -> float:

        result = 0.0

        for i in range(len(args) - 1):
            x = args[i]
            y = args[i + 1]
            result += (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        return result

    def diff(self, args: np.array) -> np.array:

        n = len(args)
        grad = np.zeros(n)

        for i in range(n):

            if i == 0:
                grad[i] = -2 * (1 - args[i]) - 400 * args[i] * (args[i + 1] - args[i] ** 2)

            elif i == n - 1:
                grad[i] = 200 * (args[i] - args[i - 1] ** 2)

            else:
                grad[i] = (
                    200 * (args[i] - args[i - 1] ** 2)
                    - 2 * (1 - args[i])
                    - 400 * args[i] * (args[i + 1] - args[i] ** 2)
                )

        return grad

    def doiff(self, args: np.array) -> np.array:
        """
        Hessiana solo implementada correctamente para 2 dimensiones.
        """

        x = args[0]
        y = args[1]

        dxdx = 1200 * x**2 - 400 * y + 2
        dxdy = -400 * x
        dydx = -400 * x
        dydy = 200

        return np.array([
            [dxdx, dxdy],
            [dydx, dydy]
        ])
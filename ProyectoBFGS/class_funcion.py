from funcion import funcion
import numpy as np


class funcion_imagen(funcion):

    def __init__(self, img_original, img_distorsionada):
        self.I0 = img_original.astype(float)
        self.Iu = img_distorsionada.astype(float)
        self.rows, self.cols = self.I0.shape

        # Cris: Coordenadas (x, y) de cada píxel
        y_coords, x_coords = np.mgrid[0:self.rows, 0:self.cols]
        self.x_coords = x_coords.astype(float)
        self.y_coords = y_coords.astype(float)

    def _transformar(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(6,)
        xp = theta[0] * self.x_coords + theta[1] * self.y_coords + theta[2]
        yp = theta[3] * self.x_coords + theta[4] * self.y_coords + theta[5]
        return xp, yp

    def _interpolar_bilineal(self, img, xp, yp):
        rows, cols = img.shape
        x0 = np.floor(xp).astype(int)
        y0 = np.floor(yp).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        # Clamping: bordes se extienden
        x0c = np.clip(x0, 0, cols - 1)
        y0c = np.clip(y0, 0, rows - 1)
        x1c = np.clip(x1, 0, cols - 1)
        y1c = np.clip(y1, 0, rows - 1)

        wx = xp - np.floor(xp)
        wy = yp - np.floor(yp)

        result = ((1 - wx) * (1 - wy) * img[y0c, x0c] +
                  wx * (1 - wy) * img[y0c, x1c] +
                  (1 - wx) * wy * img[y1c, x0c] +
                  wx * wy * img[y1c, x1c])
        return result

    def eval(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(6,)
        xp, yp = self._transformar(theta)
        Iu_t = self._interpolar_bilineal(self.Iu, xp, yp)
        diff = self.I0 - Iu_t
        return np.sum(diff ** 2)

    def diff(self, theta):
        # Gradiente por diferencias finitas centrales.
        # Con solo 6 parámetros
        theta = np.asarray(theta, dtype=float).reshape(6,)
        grad = np.zeros(6)
        h = 1e-5
        for i in range(6):
            tp = theta.copy()
            tm = theta.copy()
            tp[i] += h
            tm[i] -= h
            grad[i] = (self.eval(tp) - self.eval(tm)) / (2.0 * h)
        return grad

    def doif(self, theta, tol=1e-6, max_iter=100):
        theta = np.asarray(theta, dtype=float).reshape(6,)
        n = 6
        H = np.eye(n)
        g = self.diff(theta)

        k = 0
        while np.linalg.norm(g) > tol and k < max_iter:
            p = -H @ g
            theta_nuevo = theta + p
            g_nuevo = self.diff(theta_nuevo)

            s = theta_nuevo - theta
            y = g_nuevo - g
            ys = np.dot(y, s)

            if ys != 0:
                rho = 1.0 / ys
                I = np.eye(n)
                H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

            theta = theta_nuevo
            g = g_nuevo
            k += 1

        return theta
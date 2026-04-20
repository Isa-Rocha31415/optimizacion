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
        xp, yp = self._transformar(theta)
        Iu_t = self._interpolar_bilineal(self.Iu, xp, yp)
        diff = self.I0 - Iu_t
        return np.sum(diff ** 2)

    def diff(self, theta):
        #Gradiente por diferencias finitas centrales.
        #Con solo 6 parámetros
        grad = np.zeros(6)
        h = 1e-5
        for i in range(6):
            tp = theta.copy(); tp[i] += h
            tm = theta.copy(); tm[i] -= h
            grad[i] = (self.eval(tp) - self.eval(tm)) / (2.0 * h)
        return grad
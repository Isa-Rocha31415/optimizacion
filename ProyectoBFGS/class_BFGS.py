
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio.v2 as imageio
from class_funcion import *
from class_BFGS import *
class FuncionImagen:
    def __init__(self, img_t, img_obj, h=5e-4):
        self.img_t = img_t
        self.img_obj = img_obj
        self.h = h
        self.historial = []
        self._eval_cache = {}
        self._grad_cache = {}

    def _make_key(self, x):
        return tuple(np.round(x, 10))

    def eval(self, x):
        key = self._make_key(x)
        if key in self._eval_cache:
            val = self._eval_cache[key]
        else:
            val = funcion_objetivo(x, self.img_t, self.img_obj)
            self._eval_cache[key] = val
        self.historial.append(val)
        return val

    def grad(self, x):
        key = self._make_key(x)
        if key in self._grad_cache:
            return self._grad_cache[key].copy()

        g = np.zeros_like(x)
        f0 = self.eval(x)

        for i in range(len(x)):
            xh = np.array(x, float)
            xh[i] += self.h
            g[i] = (self.eval(xh) - f0) / self.h

        self._grad_cache[key] = g.copy()
        return g

def bfgs(func, x0, tau=1e-7, max_iter=150):
    x = np.array(x0, float)
    n = len(x)
    H = np.eye(n)
    g = func.grad(x)

    for k in range(max_iter):
        if np.linalg.norm(g) < tau:
            break

        p = -H @ g

        if np.dot(p, g) >= -1e-10:
            print(f"[BFGS] Reiniciando H en iteración {k}")
            p = -g
            H = np.eye(n)

        alpha = 1.0
        f0 = func.historial[-1]
        c1 = 0.0001
        slope = np.dot(g, p)

        for _ in range(30):
            x_trial = x + alpha * p
            f_trial = func.eval(x_trial)
            if f_trial <= f0 + c1 * alpha * slope:
                break
            alpha *= 0.8
            if alpha < 1e-10:
                alpha = 1e-4
                break

        s = alpha * p
        x_new = x + s
        g_new = func.grad(x_new)
        y = g_new - g

        sy = np.dot(s, y)

        if sy > 1e-8 * np.linalg.norm(s) * np.linalg.norm(y):
            rho = 1.0 / sy
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

            try:
                eigvals = np.linalg.eigvalsh(H)
                min_eig = np.min(eigvals)
                if min_eig < 1e-8:
                    H = H + (1e-6 - min_eig) * np.eye(n)
            except:
                print("[BFGS] Error en eigenvalues, reiniciando H")
                H = np.eye(n)
        else:
            if k % 10 == 0:
                print(f"[BFGS] Curvatura baja en iter {k}: sy={sy:.2e}")

        x = x_new
        g = g_new

    return x
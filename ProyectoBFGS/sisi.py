
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio.v2 as imageio

def to_float01(img):
    arr = np.asarray(img, dtype=float)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[...,0], arr[...,1], arr[...,2]
        arr = 0.299*r + 0.587*g + 0.114*b
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr, dtype=float)
    return arr.astype(float)

def transformar_matriz(matriz, t1, t2, t3, t4, t5, t6):
    alto, ancho = matriz.shape
    cx, cy = ancho / 2.0, alto / 2.0

    det = t1 * t4 - t2 * t3
    if abs(det) < 1e-10:
        return np.zeros_like(matriz)

    inv_t1 = t4 / det
    inv_t2 = -t2 / det
    inv_t3 = -t3 / det
    inv_t4 = t1 / det

    inv_t5 = -(inv_t1 * t5 + inv_t2 * t6)
    inv_t6 = -(inv_t3 * t5 + inv_t4 * t6)

    y, x = np.mgrid[0:alto, 0:ancho]
    x_c = x - cx
    y_c = y - cy

    xs = inv_t1 * x_c + inv_t2 * y_c + cx + inv_t5
    ys = inv_t3 * x_c + inv_t4 * y_c + cy + inv_t6

    return ndimage.map_coordinates(matriz, [ys, xs], order=3, mode='constant', cval=0)

def error_cuadratico(a, b):
    return np.sum((a - b) ** 2) / (2 * a.size)

def funcion_objetivo(params, img_t, img_obj):
    rec = transformar_matriz(img_t, *params)
    return error_cuadratico(rec, img_obj)

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

# ===== MAIN =====

I1 = to_float01(imageio.imread("I_1.pgm"))
I6 = to_float01(imageio.imread("I_6.pgm"))

func = FuncionImagen(I6, I1)
x0 = [1,0,0,1,0,0]

sol = bfgs(func, x0)
rec = transformar_matriz(I6, *sol)

print("Parámetros:", sol)
print("Error final:", error_cuadratico(rec, I1))

error_val = error_cuadratico(rec, I1)
# Formateamos los parámetros a 4 decimales para que quepan bien en el gráfico
params_str = ", ".join([f"{val:.4f}" for val in sol])

# 2. Creamos el mensaje unificado
info_pantalla = f"Parámetros: [{params_str}]\nError final: {error_val:.8f}"

# 3. El "Check" en consola (imprimirá exactamente lo mismo que el gráfico)
print("-" * 20)
print(info_pantalla)
print("-" * 20)

# ===== VISUAL =====
plt.figure(figsize=(12, 6))

plt.subplot(1,3,1)
plt.imshow(I1, cmap='gray', vmin=0, vmax=1)
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(I6, cmap='gray', vmin=0, vmax=1)
plt.title("Imagen a transformar")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(rec, cmap='gray', vmin=0, vmax=1)
plt.title("Resultado BFGS")
plt.axis('off')

# Usamos la misma variable 'info_pantalla' para el gráfico
plt.figtext(0.5, 0.05, info_pantalla, ha="center", fontsize=10, 
            family='monospace', bbox={"facecolor":"lightgray", "alpha":0.2, "pad":10})

plt.tight_layout(rect=[0, 0.15, 1, 0.95])
plt.show()
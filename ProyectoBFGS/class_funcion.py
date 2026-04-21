import numpy as np
from funcion import funcion

class funcion_imagen(funcion):
    def __init__(self, img_original, img_distorsionada):
        self.I0 = img_original.astype(float)
        self.Iu = img_distorsionada.astype(float)
        self.rows, self.cols = self.I0.shape
        
        # Coordenadas (x, y) -> (columna, fila)
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
        x1, y1 = x0 + 1, y0 + 1

        # Máscara de validez (EVITA que el optimizador "caiga" en bordes planos)
        valid = (x0 >= 0) & (x1 < cols) & (y0 >= 0) & (y1 < rows)
        
        # Safe indexing
        x0c, y0c = np.clip(x0, 0, cols-1), np.clip(y0, 0, rows-1)
        x1c, y1c = np.clip(x1, 0, cols-1), np.clip(y1, 0, rows-1)

        wx = xp - np.floor(xp)
        wy = yp - np.floor(yp)

        I_interp = ((1 - wx) * (1 - wy) * img[y0c, x0c] +
                    wx * (1 - wy) * img[y0c, x1c] +
                    (1 - wx) * wy * img[y1c, x0c] +
                    wx * wy * img[y1c, x1c])
        
        # Retornamos también la máscara para usarla en eval/diff
        return I_interp, valid

    def transformacion_prueba(coords, theta):
            """
        Aplica una transformación afín de 6 parámetros a un punto (x, y).
        
        Args:
            coords (tuple): Un par (x, y) con las coordenadas originales.
            theta (list/array): Un arreglo o lista con los 6 parámetros [t1, t2, t3, t4, t5, t6].
            
        Returns:
            tuple: Las nuevas coordenadas (x_prim, y_prim).
        """
        # Desempaquetamos las coordenadas originales
        xn, yn = coords
        
        # Desempaquetamos los parámetros theta para mayor claridad
        # theta[0]=t1, theta[1]=t2, theta[2]=t3 (traslación x)
        # theta[3]=t4, theta[4]=t5, theta[5]=t6 (traslación y)
        t1, t2, t3, t4, t5, t6 = theta
        
        # Aplicamos la multiplicación de matrices simplificada
        x_prim = t1 * xn + t2 * yn + t3
        y_prim = t4 * xn + t5 * yn + t6
        
        return (x_prim, y_prim)

# --- Ejemplo de uso ---
puntos_originales = (10, 20)
parametros = [1.2, 0.2, 5.0, 0.1, 1.1, -3.0]

resultado = transformacion_afin(puntos_originales, parametros)
print(f"Coordenadas transformadas: {resultado}")        

    
    def eval(self, theta):
        theta = np.asarray(theta, dtype=float)
        xp, yp = self._transformar(theta)
        Iu_t, valid = self._interpolar_bilineal(self.Iu, xp, yp)
        
        # Solo evaluamos píxeles que están dentro de la imagen
        diff = (self.I0 - Iu_t)[valid]
        # Normalización por número de píxeles válidos (escala ~1)
        return np.sum(diff ** 2) / np.maximum(1.0, np.sum(valid))
        
        # Sumatoria de errores al cuadrado pura
        return np.sum(diff ** 2)
    def diff(self, theta):
        theta = np.asarray(theta, dtype=float)
        grad = np.zeros(6)
        # Paso relativo: mejor para parámetros de traslación (pixeles) vs rotación (adimensionales)
        h = np.clip(np.abs(theta) * 1e-3, 1e-4, 1e-3)
        
        for i in range(6):
            tp, tm = theta.copy(), theta.copy()
            tp[i] += h[i]
            tm[i] -= h[i]
            grad[i] = (self.eval(tp) - self.eval(tm)) / (2.0 * h[i])
        print("diff hizo su chamba")
        return grad

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio.v2 as imageio

I1 = to_float01(imageio.imread("I_1.pgm"))
I6 = to_float01(imageio.imread("I_6.pgm"))

func = FuncionImagen(I6, I1)
x0 = [1,0,0,1,0,0]

sol = bfgs(func, x0)
rec = transformar_matriz(I6, *sol)

print("Parámetros:", sol)
print("Error final:", error_cuadratico(rec, I1))

# ===== VISUAL =====

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(I1, cmap='gray', vmin=0, vmax=1)
plt.title("Objetivo")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(I6, cmap='gray', vmin=0, vmax=1)
plt.title("Transformada")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(rec, cmap='gray', vmin=0, vmax=1)
plt.title("BFGS")
plt.axis('off')

plt.tight_layout()
plt.show()
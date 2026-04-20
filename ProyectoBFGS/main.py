from class_BFGS import BFGS
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
from class_funcion import funcion
###########################################################
# 4. Cris (Transformation Logic): Modelo de 6 Parámetros  #
###########################################################

img_original = r"./ProyectoBFGS/img/I_1.pgm" 
img_chueca = r"./ProyectoBFGS/img/I_6.pgm" 

if __name__ == "__main__" : 
    # Leer la imagen con Pillow 
    img_original = np.array(Image.open(img_original)).astype(float)
    img_chueca = np.array(Image.open(img_chueca)).astype(float)
    
    func : funcion = funcion(img_chueca) 
    model : BFGS = BFGS(func)  

    # Tratando el arreglo de thetas inicial (0, 0, 0, 0, 0, 0) por pobar. 
    thetas = model.solve(np.array([1, 1, 1, 1, 1, 1]))
from class_BFGS import BFGS
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

###########################################################
# 4. Cris (Transformation Logic): Modelo de 6 Parámetros  #
###########################################################

img_original = r"./ProyectoBFGS/img/I_1.pgm" 
img_chueca = r"./ProyectoBFGS/img/I_6.pgm" 

if __name__ == "__main__" : 
    # Leer la imagen con Pillow 
    img_original = np.array(Image.open(img_original)).astype(float)
    img_chueca = np.array(Image.open(img_chueca)).astype(float)
    
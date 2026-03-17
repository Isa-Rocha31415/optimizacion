from class_GC import GC
from class_funcion import funcion as softened 
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

img_path = r"./ProyectoGradienteConjugado/img/lena.ascii.pgm" 

if __name__ == "__main__" : 

    

    # Leer la imagen con Pillow 
    img = np.array(Image.open(img_path)) 
    
    # Guardar su forma para luego
    img_shape = img.shape

    # Configuración del oprimizador y ejecutar
    objective_function = softened(img, lamda=0.5) 
    algorithm = GC(objective_function) 
    softened_image = algorithm.solve(img)

    # regresar a la forma original
    softened_image = np.reshape(softened_image, img_shape)

    # mostrar
    plt.imshow(softened_image, cmap='gray')
    plt.show()
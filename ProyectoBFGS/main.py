from class_BFGS import BFGS
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
from class_funcion import funcion_imagen
###########################################################
# 4. Cris (Transformation Logic): Modelo de 6 Parámetros  #
###########################################################

img_o = r"I_1.pgm" 
img_c = r"I_6.pgm" 

if __name__ == "__main__" : 
    # Leer la imagen con Pillow 
    img_original = np.array(Image.open(img_o)).astype(float) /255.0
    img_chueca = np.array(Image.open(img_c)).astype(float) /255.0
    
    func = funcion_imagen(img_original,img_chueca) 
    model = BFGS(func)  

    #Tratando el arreglo de thetas inicial (0, 0, 0, 0, 0, 0) por pobar. 
    theta_inicial = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    print("Iniciando optimización BFGS...")
    
    # --- 3. Ejecutar la optimización ---
    # Esto buscará el theta* que minimice f(theta)
    theta_optimo = model.solve(theta_inicial)
    
    print(f"Optimización finalizada.\nTheta óptimo: {theta_optimo}")
    
    # --- 4. Generar la imagen final (Mapeo Inverso) ---
    # Usamos los métodos de tu clase para reconstruir la imagen con el theta hallado
    xp, yp = func._transformar(theta_optimo)
    img_rectificada = func._interpolar_bilineal(img_chueca, xp, yp)
    
    # --- Visualización de Resultados ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title("Original")
    
    axes[1].imshow(img_chueca, cmap='gray')
    axes[1].set_title("Distorsionada")
    
    axes[2].imshow(img_rectificada)
    axes[2].set_title("Reconstruida ")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    
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

    theta_inicial = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    #theta_inicial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print("Iniciando optimización BFGS...")
    
    # --- 3. Ejecutar la optimización ---
    # Esto buscará el theta* que minimice f(theta)
    theta_optimo = model.solve(theta_inicial)
    
    print(f"Optimización finalizada.\nTheta óptimo: {theta_optimo}")
    
    # --- 4. Generar la imagen final (Mapeo Inverso) ---
    # Usamos los métodos de tu clase para reconstruir la imagen con el theta hallado
    xp, yp = func._transformar(theta_optimo)
    img_rectificada, _ = func._interpolar_bilineal(img_chueca, xp, yp)
    
    # --- 5. Cálculo de Error ---
    # Diferencia absoluta punto a punto
    error_map = np.abs(img_original - img_rectificada)
    
    # Métricas numéricas
    sse = np.sum((img_original - img_rectificada)**2) # Sum of Squared Errors
    mse = np.mean((img_original - img_rectificada)**2) # Mean Squared Error
    
    print(f"\n--- Métricas de Calidad ---")
    print(f"Suma de Errores al Cuadrado (SSE): {sse:.4f}")
    print(f"Error Cuadrático Medio (MSE): {mse:.6e}")

    # --- 6. Visualización Extendida ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5)) # Cambiamos a 4 columnas
    
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title("Original ($I_0$)")
    
    axes[1].imshow(img_chueca, cmap='gray')
    axes[1].set_title("Distorsionada ($I_u$)")
    
    axes[2].imshow(img_rectificada, cmap='gray')
    axes[2].set_title("Reconstruida ($I_r$)")
    
    # El mapa de calor del error
    # Usamos 'hot' para que las diferencias grandes se vean en rojo/amarillo
    im_err = axes[3].imshow(error_map, cmap='hot')
    axes[3].set_title("Mapa de Error (Diferencia)")
    fig.colorbar(im_err, ax=axes[3], fraction=0.046, pad=0.04)
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
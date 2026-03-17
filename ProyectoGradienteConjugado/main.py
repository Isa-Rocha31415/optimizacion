from class_GC import GC
from class_funcion import funcion as softened 
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

img_path = r"./ProyectoGradienteConjugado/img/lena.ascii.pgm" 

if __name__ == "__main__" : 
    # Leer la imagen con Pillow 
    img = np.array(Image.open(img_path)).astype(float)
    img_norm = img /255.0
    img_shape =img_norm.shape
    
    lambdas_to_test = [0.01, 0.1, 1.0, 10.0,16.0,32.0,64.0]
    fig, axes = plt.subplots(1, len(lambdas_to_test) + 1, figsize=(20, 5))

    # Mostrar la original primero
    axes[0].imshow(img_norm, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i, lmd in enumerate(lambdas_to_test):
        print(f"Procesando Lambda = {lmd}...")
        
        # Instanciar con el lambda actual
        obj_func = softened(img_norm, lamda=lmd)
        algorithm = GC(obj_func)
        
        # Resolvemos (Pasamos la imagen plana como punto inicial)
        result_flat = algorithm.solve(img_norm.flatten())
        
        # Reconstruir y mostrar
        img_out = np.reshape(result_flat, img_shape)
        
        axes[i+1].imshow(img_out, cmap='gray', vmin=0, vmax=1)
        axes[i+1].set_title(f"λ = {lmd}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig("./ProyectoGradienteConjugado/img/evidencia_lambdas.png", dpi=300)
    print("\n¡Listo! Comparativa guardada en 'img/evidencia_lambdas.png'")
    plt.show()
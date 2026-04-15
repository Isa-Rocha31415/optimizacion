"""
FUNCIÓN DE COSTO PARA SUAVIZADO DE IMÁGENES (Regularización de Tikhonov)
-----------------------------------------------------------------------
E(X) = Σ [ (X_i,j - I_i,j)^2 + λ * ( (X_i,j - X_i+1,j)^2 + 
                                     (X_i,j - X_i-1,j)^2 + 
                                     (X_i,j - X_i,j+1)^2 + 
                                     (X_i,j - X_i,j-1)^2 ) ]

Donde:
- (X_i,j - I_i,j)^2 : Término de FIDELIDAD (mantiene la imagen cerca de la original).
- λ (lambda)        : PARÁMETRO DE REGULARIZACIÓN (controla la fuerza del suavizado).
- Términos vecinos  : PENALIZACIÓN POR DIFERENCIAS (suaviza el ruido).

GRADIENTE LOCAL (para optimización):
∂E/∂X_i,j = 2(X_i,j - I_i,j) + 2*λ * [ 4*X_i,j - (X_i+1,j + X_i-1,j + X_i,j+1 + X_i,j-1) ]

# Clase donde va la funcion
# Cris Gradiente
# Tadeo Tadeo
# Isa  Eval

"""


from funcion import funcion
import numpy as np

# Vamos a utilizar condiciones de barrera- Toroide

class funcion(funcion):
    
    def __init__(self, img, lamda =0.1):
        #imagen
        self.img_original = img.astype(float)
        self.lamda =lamda
        self.rows, self.cols = img.shape



    def eval(self, x_flat):
        """
        Método que evalúa la función de suavisado 
        utilizamos la condicion de barrera (toroide)
        recorre toda la imagen  y lo hace en forma de toroide
        x_flat: la imagen actual (X) convertida en un vector unidimensional.
        
        """
        # recontruir la matriz img desde el vector plano
        X = x_flat.reshape((self.rows, self.cols))
        I = self.img_original

        # termino de fidelidad
        fidelidad = np.sum((X-I)**2)

        # Termino de suavizado con condicion de Toroide
        # calculamos la diferencia con los vecinos desplzando la matrix completa

        dif_derecha = (X- np.roll(X, shift=-1, axis =1))**2 #X_ij - X_i,j+1
        dif_izq = (X -np.roll(X, shift =1, axis=1))**2  # X_ij - X_i,j-1
        dif_abajo = (X- np.roll(X, shift=-1, axis =0))**2 # X_ij - X_i+1,j
        dif_arriba = (X -np.roll(X, shift=1, axis =0))**2  # X_ij - X_i-1,
        
        suavizado = np.sum(dif_derecha + dif_izq +dif_abajo + dif_arriba)
        return  fidelidad + (self.lamda * suavizado)
    
        # @Cris: He actualizado diff para usar el Gradiente Analítico.
    # El método anterior (diferencias finitas con h) era O(n), lo que 
    # hacía que procesar una imagen pequeña tardara minutos. 
    # Con np.roll calculamos el gradiente de toda la imagen en un solo paso 
    # matemático, aprovechando que ya conocemos la derivada de la función.
    def diff(self, x_flat):
        X = x_flat.reshape((self.rows, self.cols))
        I = self.img_original
        
        # Derivada analítica: 2(X - I) + 2*λ*(4X - suma_vecinos)
        # Usamos np.roll para cumplir la condición de Toroide de forma vectorial
        suma_vecinos = (np.roll(X, -1, 1) + np.roll(X, 1, 1) + 
                        np.roll(X, -1, 0) + np.roll(X, 1, 0))
        
        grad = 2 * (X - I) + 2 * self.lamda * (4 * X - suma_vecinos)
        return grad.flatten()
        

    # @Tadeo: He modificado la Hessiana. Calcular la matriz completa con h es ineficiente
    # devolvemos el resultado de A * vec_dirrecion
    #utilizamos numpy por eficiencia como si fuera escrito en C
    def doif(self, x_flat):
        vec_direccion = x_flat.reshape((self.rows, self.cols))
        # 2. La Hessiana de nuestra función aplicada a un vector d es:
        suma_vecinos_vec = (np.roll(vec_direccion, -1, 1) + np.roll(vec_direccion, 1, 1) + 
                        np.roll(vec_direccion, -1, 0) + np.roll(vec_direccion, 1, 0))

        resultado =2 * vec_direccion +2 *self.lamda *(4 * vec_direccion - suma_vecinos_vec)  
        return resultado.flatten()     
"""
FUNCIÓN OBJETIVO: f(theta) - Registro de Imágenes
Referencia: func_objetivo.jpeg y Optimization Course 1.1

ESTA FUNCIÓN CUANTIFICA EL DESEMPEÑO DE LOS PARÁMETROS DE OPTIMIZACIÓN [1].

1. FORMULACIÓN MATEMÁTICA:
   f(theta) = || I_0 - I_u(theta) ||^2_2
   f(theta) = Sum_{x=1}^n Sum_{y=1}^m (P^0_{x,y} - P^{u*}_{x,y})^2

2. SIGNIFICADO TÉCNICO:
   - Es una función de MÍNIMOS CUADRADOS que mide el error de intensidad píxel a píxel.
   - P^0_{x,y}: Intensidad del píxel en la imagen original (referencia).
   - P^{u*}_{x,y}: Intensidad del píxel en la imagen transformada tras aplicar el 
     vector de 6 parámetros theta.
   - theta: Vector de variables en DOMINIO REAL (Real-domain coding), común en la 
     optimización paramétrica de diseños.

3. LÓGICA DE OPERACIÓN:
   a. Recibe un vector theta con 6 valores (rotación, sisaña, traslación).
   b. Transforma la imagen "chueca" (I_u) usando estos parámetros.
   c. Calcula la diferencia de intensidad entre cada píxel de la imagen resultante 
      y la original.
   d. Eleva las diferencias al cuadrado y las suma para obtener un escalar real [1, 2].

4. OBJETIVO DEL ALGORITMO (BFGS):
   - BFGS buscará el vector theta que minimice esta función (f(theta) -> 0) [4, 5].
   - Al encontrar el MÍNIMO GLOBAL, las imágenes serán prácticamente idénticas, 
     lo que significa que la rotación y sisaña han sido corregidas con éxito [6].
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
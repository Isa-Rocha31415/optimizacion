# Suavizado de Imágenes mediante Regularización de Tikhonov

Este proyecto implementa un algoritmo de procesamiento digital de imágenes para la reducción de ruido (denoising). Utilizamos un enfoque de optimización basado en la **Regularización de Tikhonov**, que permite suavizar una imagen manteniendo la fidelidad con la estructura original.

## 1. Introducción al Problema

El objetivo es encontrar una imagen ideal $X$ que minimice el ruido presente en una imagen observada $I$. Para ello, se define una función de energía o costo que equilibra dos fuerzas opuestas: la similitud con la imagen original y la suavidad entre píxeles adyacentes.

La función objetivo implementada es:

$$E(X) = \sum_{i,j} \left( (X_{i,j} - I_{i,j})^2 + \lambda \left[ (X_{i,j} - X_{i+1,j})^2 + (X_{i,j} - X_{i-1,j})^2 + (X_{i,j} - X_{i,j+1})^2 + (X_{i,j} - X_{i,j-1})^2 \right] \right)$$

* **Término de Fidelidad $(X_{i,j} - I_{i,j})^2$:** Evita que la solución se aleje demasiado de los datos originales.
* **Término de Suavizado ($\lambda$):** Penaliza las variaciones bruscas de intensidad entre un píxel y sus cuatro vecinos inmediatos.
* **Archivo clave:** Toda la lógica matemática de esta función, incluyendo el cálculo del **Gradiente** y la **Hessiana**, se encuentra centralizada en `class_function.py`.



---

## 2. Estructura de Archivos

| Archivo | Descripción |
| :--- | :--- |
| **`class_function.py`** | Define la clase principal con la función de costo, el gradiente y la Hessiana del problema. |
| **`class_GC.py`** | Contiene la implementación del algoritmo de **Gradiente Conjugado** para la optimización. |
| **`funciones_test.py`** | Pruebas unitarias para validar que los cálculos de derivadas sean correctos. |
| **`funcion.py`** | Scripts auxiliares para el manejo de estructuras de datos y transformaciones. |
| **`img/`** | Directorio destinado a almacenar las imágenes de entrada y los resultados procesados. |
| **`main.py`** | Orquestador del proyecto: carga la imagen, configura parámetros y ejecuta el suavizado. |
| **`readme.md`** | Documentación del proyecto. |

---

## 3. Funcionamiento del Código

El sistema funciona de manera modular para garantizar escalabilidad:

1.  **Carga y Preprocesamiento:** El archivo `main.py` lee una imagen ruidosa desde la carpeta `img/`.
2.  **Configuración de Hiperparámetros:** Se define el valor de $\lambda$ (lambda). Este valor es crítico, ya que determina qué tan "agresivo" será el filtro de suavizado.
3.  **Ensamblado:** En el `main`, se instancia la clase de `class_function.py` y se vincula con el optimizador definido en `class_GC.py`. 
4.  **Optimización:** El algoritmo de Gradiente Conjugado utiliza las derivadas (gradiente y hessiana) para iterar sobre la imagen hasta encontrar el mínimo de la función de energía.
5.  **Salida:** Se reconstruye la imagen a partir del vector optimizado y se guarda el resultado comparativo.

---

## 4. Evidencia y Comparativa de Resultados

La evidencia de desempeño de este proyecto se basa en la comparación visual y cuantitativa de la imagen original frente a la procesada. 

### Análisis de los diferentes Lambdas ($\lambda$)
Es obligatorio realizar pruebas con distintos valores de $\lambda$ para observar su efecto en la imagen:

* **$\lambda$ Pequeño (ej. 0.05):** El suavizado es ligero; conserva los detalles finos pero puede dejar ruido residual.
* **$\lambda$ Grande (ej. 2.0):** El ruido desaparece por completo, pero la imagen sufre un efecto de desenfoque (blur) y pierde definición en los bordes.

> **Nota:** La selección del $\lambda$ óptimo depende del nivel de ruido inicial en la imagen. En el reporte final se deben incluir las imágenes resultantes para al menos tres valores distintos de lambda.
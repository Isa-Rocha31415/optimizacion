# Proyecto de Registro de Imágenes: Algoritmo BFGS

Este documento define la organización y las especificaciones técnicas para el equipo de desarrollo encargado de resolver el problema de alineación de imágenes mediante optimización paramétrica.

---

## 1. Isabel (Scrum Master): Coordinación e Integración
**Tarea Principal:** Supervisar el flujo de trabajo y la integración final en el archivo `main.py`.
**Responsabilidades:**
*   Asegurar que se cumplan los **criterios de parada** del algoritmo: la tolerancia del gradiente ($||g|| < \tau$) y el límite de iteraciones ($k < k_{max}$).
*   Integrar los módulos de la función objetivo, la transformación de imagen y el optimizador BFGS.
*   Gestionar el repositorio para que las pruebas en `funciones_test` sean consistentes.

---

## 2. Rodolfo y July (Algorithms Team): Optimizador BFGS
**Tarea Principal:** Desarrollar el núcleo del algoritmo BFGS (Broyden-Fletcher-Goldfarb-Shannon).

### Explicación del Algoritmo BFGS (Basado en la imagen)
A diferencia del Gradiente Conjugado que ya tienen de base, el BFGS es un método **Quasi-Newton**. Su lógica según su pizarrón es:
1.  **Inicialización:** Definir $x_0$, $H_0$ (normalmente la Identidad $I$) y calcular el gradiente inicial $g_0$.
2.  **Dirección de Búsqueda ($p_k$):** Se calcula como $p_k = -H_k g_k$. Aquí $H_k$ es la aproximación de la **inversa del Hesiano**.
3.  **Búsqueda de Línea (July):** Encontrar $\alpha_k$ tal que $f(x_k + \alpha_k p_k)$ disminuya. Deben implementar condiciones como **Wolfe**, **Armijo** o **Goldbach** (Goldstein) para asegurar la convergencia.
4.  **Actualización de Curvatura:**
    *   $s_k = x_{k+1} - x_k$ (Cambio en posición).
    *   $y_k = g_{k+1} - g_k$ (Cambio en gradiente).
    *   $\rho_k = \frac{1}{y_k^T s_k}$.
5.  **Fórmula de actualización de $H_{k+1}$:** Es la fórmula larga al final de la imagen que actualiza la matriz sin invertirla explícitamente, reduciendo la complejidad de $O(n^3)$ a $O(n^2)$.

### Comparación con Gradiente Conjugado (GC)
| Característica | Gradiente Conjugado (Su código base) | BFGS (Su nueva tarea) |
| :--- | :--- | :--- |
| **Dirección** | Basada en direcciones $A$-conjugadas y el parámetro $\beta$. | Basada en la matriz $H_k$ que imita al método de Newton. |
| **Memoria** | Muy eficiente, no guarda matrices grandes. | Guarda una matriz $H$ de $n \times n$ (en su caso es $6 \times 6$). |
| **Uso** | Ideal para sistemas lineales grandes y formas cuadráticas. | El más popular y efectivo para problemas no lineales generales. |
| **Complejidad** | Depende del producto matriz-vector $Ap$. | Evita calcular el Hesiano real, usando actualizaciones de rango 2. |

---

## 3. Mateo (Function Logic): Implementación de la Función Objetivo
**Tarea Principal:** Crear una clase que herede de `funcion` para el problema de imágenes.
**Responsabilidades:**
*   Implementar `eval(args)` para calcular $f(\theta)$.
*   Implementar `diff()` para el cálculo del gradiente $\nabla f(\theta)$.

### Explicación de la Función Objetivo
La función que debes programar es:
$$f(\theta) = \sum_{x=1}^n \sum_{y=1}^m (P_{x,y}^0 - P_{x,y}^{u^*})^2$$
*   **Significado:** Es una suma de errores al cuadrado. Cuantifica qué tan diferente es la imagen original ($P^0$) de la imagen transformada ($P^{u^*}$).
*   **Meta:** El optimizador (Rodolfo/July) buscará los $\theta$ que minimicen este valor hasta que las imágenes coincidan.

---

## 4. Cris (Transformation Logic): Modelo de 6 Parámetros
**Tarea Principal:** Implementar la transformación afín que "endereza" la imagen.
**Responsabilidades:**
*   Aplicar el vector $\theta$ de 6 parámetros a cada píxel de la imagen.
*   Garantizar que la reconstrucción de la imagen sea precisa para que Mateo pueda evaluarla.


### Explicación del Modelo
Usarás una **codificación en el dominio real** para los 6 parámetros. La fórmula proyecta cada coordenada $(x_n, y_n)$ a una nueva posición $(x', y')$ de la siguiente manera:
1.  $x' = \theta_1 x_n + \theta_2 y_n + \theta_3$
2.  $y' = \theta_4 x_n + \theta_5 y_n + \theta_6$

Donde $\theta_3$ y $\theta_6$ controlan el desplazamiento (traslación), y el resto controla la rotación y la sisaña (shear) que deben corregir.
## 2. El Modelo de los 6 Parámetros ($\theta$)

Para corregir la rotación y la sisaña (shear), utilizamos una codificación en el dominio real, común en la optimización paramétrica de diseños de ingeniería. La fórmula consiste en una **transformación afín** que mapea cada píxel original $(x_n, y_n)$ a nuevas coordenadas $(x', y')$ mediante el vector de parámetros $\theta \in \mathbb{R}^6$:

$$
\begin{bmatrix}
x_n & y_n & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & x_n & y_n & 1
\end{bmatrix}
\begin{bmatrix}
\theta_1 \\
\theta_2 \\
\theta_3 \\
\theta_4 \\
\theta_5 \\
\theta_6
\end{bmatrix}
=
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
$$



Esta operación se traduce en dos ecuaciones lineales que definen la nueva posición del píxel:

* $x' = \theta_1 x_n + \theta_2 y_n + \theta_3$
* $y' = \theta_4 x_n + \theta_5 y_n + \theta_6$

### ¿Qué controla cada parámetro $\theta$?

* **$\theta_3$ y $\theta_6$**: Representan la **traslación**, moviendo la imagen en los ejes horizontal y vertical respectivamente.
* **$\theta_1, \theta_2, \theta_4, \theta_5$**: Controlan de forma combinada la **rotación**, el **escalado** y la **sisaña (shear)**, permitiendo estirar o inclinar la imagen para compensar la distorsión.

---

## 3. ¿Cómo trabaja BFGS en este problema?

En este contexto, el algoritmo **BFGS** (Broyden–Fletcher–Goldfarb–Shanno) no busca coordenadas espaciales, sino que su "espacio de búsqueda" es el vector de parámetros óptimos $\theta \in \mathbb{R}^6$.

BFGS es un método **Quasi-Newton** que destaca por reducir la complejidad computacional de $O(n^3)$ a $O(n^2)$ al no requerir el cálculo directo del Hesiano, sino una aproximación iterativa.

1.  **Inicialización**: Se comienza con un punto inicial $\theta_0$ (normalmente la matriz identidad: $\theta = [1, 0, 0, 0, 1, 0]$) y una aproximación inicial de la inversa del Hesiano $H_0$ (usualmente la matriz identidad).
2.  **Iteración**: El algoritmo propone un nuevo vector $\theta_{k+1} = \theta_k + \alpha_k p_k$, donde $p_k$ es la dirección de descenso calculada como $-H_k \nabla f(\theta_k)$.
3.  **Evaluación**: Para cada $\theta$ propuesto, se evalúa la **función objetivo** $f(\theta)$, la cual cuantifica el error mediante la Suma de Cuadrados de las Diferencias (SSD) entre la imagen original ($P^0$) y la transformada ($P^{u*}$):
    $$f(\theta) = \sum_{x=1}^{n} \sum_{y=1}^{m} (P_{x,y}^0 - P_{x,y}^{u*})^2$$
4.  **Convergencia**: El proceso se repite hasta que el gradiente sea suficientemente pequeño ($||\nabla f_k|| < \epsilon$) o se alcance el límite de iteraciones.

---

## 4. Estructura del `main`

El programa principal integra los componentes siguiendo este flujo lógico:

1.  **Cargar las imágenes**: Lectura de la imagen original ($I_0$) y la imagen distorsionada ($I_u$).
2.  **Configurar BFGS**:
    * Definir el vector inicial $\theta_0$.
    * Establecer la función objetivo $f(\theta)$ basada en la sumatoria de errores al cuadrado.
    * Calcular o proporcionar el gradiente $\nabla f(\theta)$ respecto a los 6 parámetros.
3.  **Ejecutar la optimización**: Llamar a la rutina BFGS para obtener el vector de parámetros óptimos $\theta^*$.
4.  **Generar la imagen final**:
    * Crear un lienzo vacío del tamaño original.
    * Para cada coordenada $(x, y)$, aplicar la transformación con $\theta^*$ para determinar de qué posición de la imagen distorsionada extraer el color (**mapeo inverso**).
    * **Guardar/Mostrar el resultado**: Exportar la imagen rectificada.

> [!IMPORTANT]
> **Nota técnica**: Dado que las coordenadas calculadas $(x', y')$ suelen ser valores decimales, es fundamental implementar **interpolación bilineal** al reconstruir la imagen. Esto evita el aliasing y asegura una transición suave entre los colores de los píxeles.
---

**Nota para el equipo:** Estamos trabajando en **Python**. Asegúrense de usar `numpy` para todas las operaciones vectoriales y matriciales, siguiendo la estructura de la clase base `funcion` proporcionada.

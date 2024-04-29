# Semi-supervised Classification Trees

Este código en Python implementa árboles semi-supervisados.

## Artículo
El método está basado en la propuesta del cálculo de impureza incluido en:

- **Título:** Semi-supervised classification trees
- **Autores:** Jurica Levatić, Michelangelo Ceci, Dragi Kocev, Sašo Džeroski
- **Publicado en:** Journal of Intelligent Information Systems, 2017, Volumen 49, páginas 461-486
- **Editorial:** Springer

## SSLTree

Los árboles de clasificación semi-supervisada tienen como objetivo clasificar nuevos datos construyendo un árbol de 
decisión basado en los datos etiquetados y no etiquetados al mismo tiempo. El algoritmo construye el árbol dividiendo
iterativamente el conjunto de datos minimizando una suma ponderada de cálculos clásicos (como el coeficiente gini) y 
la variabilidad de las características.

### Clase `Node`
- Representa un nodo en el árbol de decisión.
- Contiene atributos como el subconjunto de puntos de datos, la característica de división, el valor de división, impureza y probabilidades de clase.

### Clase `SSLTree`
- Implementa el clasificador de árbol de decisión.
- Construye el árbol utilizando el cálculo de impureza propuesto por Levatić et al.
- Parámetros:
  - `w`: Controla la cantidad de supervisión.
  - `splitter`: Estrategia para elegir la división en cada nodo.
  - `max_depth`: Profundidad máxima del árbol.
  - `min_samples_split`: Número mínimo de muestras requeridas para dividir un nodo interno.
  - `min_samples_leaf`: Número mínimo de muestras requeridas para estar en un nodo hoja.
  - `max_features`: Número de características a considerar al buscar la mejor división.



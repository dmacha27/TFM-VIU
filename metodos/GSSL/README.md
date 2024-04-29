# Graph-based Semi-Supervised Learning Algorithms

Este repositorio contiene implementaciones de algoritmos de aprendizaje semi-supervisado basados en grafos en Python.

GSSL.py está organizado de la siguiente manera:

## Función `rgcli`

La función `rgcli` realiza el algoritmo de Robust Graph that Considers Labeled Instances (RGCLI). 
Construye un grafo basado en los datos de entrada y realiza etiquetado basado en la consistencia. 
El método se basa en el siguiente artículo:

- **Título:** RGCLI: Robust Graph that Considers Labeled Instances for Semi-Supervised Learning
- **Autores:** Lilian Berton, Thiago de Paulo Faleiros, Alan Valejo, Jorge Valverde-Rebaza y Alneu de Andrade Lopes
- **Publicado en:** Neurocomputing, Volume 226, Pages 238-248, 2017.
- **Disponible en:**  [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231216314680)
- **DOI:** [10.1016/j.neucom.2016.11.053](https://doi.org/10.1016/j.neucom.2016.11.053)

## Función `llgcl`

La función `llgcl` realiza la propagación de etiquetas utilizando el algoritmo de Learning with local and global consistency (LLGC). 
Propaga etiquetas a través del grafo construido por el algoritmo RGCLI (o cualquier otro grafo). 
El método se basa en el siguiente artículo:

- **Título:** Learning with local and global consistency
- **Autores:** Dengyong Zhou, Olivier Bousquet, Thomas Lal, Jason Weston y Bernhard Schölkopf
- **Publicado en:** Advances in Neural Information Processing Systems, Volume 16, 2003.

## Función `llgcl_dataset_order`

La función `llgcl_dataset_order` ordena el conjunto de datos separando las instancias etiquetadas y no etiquetadas. 
Según el artículo "Aprendizaje con Consistencia Local y Global", las primeras "l" instancias corresponden a puntos 
etiquetados, donde x_i para i<l, siendo "l" el número de instancias etiquetadas.

## Clase `GSSLTransductive`

La clase `GSSLTransductive` implementa un algoritmo de aprendizaje semi-supervisado basado en grafos en modo transductivo.

## Clase `GSSLInductive`

La clase `GSSLInductive` implementa un algoritmo de aprendizaje semi-supervisado basado en grafos en modo inductivo.



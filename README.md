# Analisis-de-datos-futbol
https://github.com/andmansim/Analisis-de-datos-futbol.git

# Graficas

Limpiamos los datos a examinar (daos_fut.csv), los analizamos (ver si hay que quitar, añadir columnas o filas, transformar alguna variable categórica en numérica, etc. ) y los representamos para ver su relación y que información nos proporcionan. 

# Regresión logística

En esta parte presento dos posible enfoques a la hora de predecir el resultado de los partidos.
En regresion.py hace el análisis con la probabilidad de ganar, donde se realiza su probabilidad enfrenada en crear_csv. Por otro lado, en regresion1.py se realiza mediante la comparación de puntos, estos se asignan dependiendo de las comparaciones de los equipos en sus capacidades ofensivas y defensivas (crear_csv1.py).

Tras ver los resultados vemos que ambos métodos presentan los mismos resultados. 


Una regresión lineal o logística se realiza para ver la relación entre las variables independientes (X) y las dependientes (Y) o para clasificar las distintas opciones o resultados. 

Diferencia entre regresión logística y lineal:
Regresion logistica se utiliza cuando la variable de respuesta es binaria. Debe de ser un número entero.
Se utiliza para clasificar, no para predecir valores continuos. Ejemplo: predecir si un correo es spam o no.

Regresion lineal es para ver la relación entre dos variables.
Se utiliza para predecir valores continuos, ejemplo: predecir el precio de una casa.

# Cluster y serie temporal

Clusterización de datos, es la división de un conjunto de datos en grupos homogéneos para buscar posibles
patrones o tendencias. Cada cluster es una agrupación. 

Es una técnica de aprendizaje no supervisado, es decir, no se le proporciona al algoritmo de 
aprendizaje ninguna información sobre las clases de los datos.

Algoritmos de clusterización:
    - K-means
    - DBSCAN
    - Mean Shift
    - Agglomerative Clustering
    - GMM 
    -OPTICS

Serie temporal, es para ver la evolución de los datos a lo largo del tiempo.
En esta serie los datos estan indexados por su orden cronológico. 
Se suele visualizar la serie temporal, la descomposición en tendencia, estacionalidad y residuo, la 
modelización de tendencias y la predicción de futuros valores.

Técnicas de análisis de series temporales:
    - ARIMA
    - SARIMA
    - AR
    - Exponential Smoothing
    - Random Forest
    - Gradient Boosting
    - Multivariate Forecasting
    - Esemble modeling

# CNN (Neuronas)
CNN sin transferencia:
Entrenamiento de redes neuronales convolucionales es un proceso de aprendizaje automático que se utiliza para aplicar filtros en regiones locales y detectar texturas, patrones, formas, etc. en datos que tienen forma de cuadrícula como las imágenes y reconocimiento de patrones. Tiene pesos y sesgos que se van ajustando según avanzamos en la red, donde las características más complejas se van fromando de las más sencillas de las capas anteriores. 

CNN con transferencia:
Aprovechamos el conocimiento adquirido por un modelo entrenado en una tarea similar a la que lo queremos evaluar. Por lo general se coge un modelo ya entrenado y se adapta a un nuevo uso más pequeño.

# DNN (Neuronas)
Entrenamiento de redes neuronales profundas es un proceso de aprendizaje automático compuesto por múltiples capas de neuronas. Consiste en ajustar pesos y sesgos de las conexiones entre las neuronas para que el modelo pueda aprender a realizar tareas específicas.  

En esta parte he hechos dos DNN:
Una encargada de clasificar si los equipos son de ataque (1), defensa (2) o neutro (3). Se encuentra en la carpeta clasificar y sus imágenes están guardadas en img. 

La otra se encarga de predecir los resltados de los partidos, siendo 1 gana el equipo local, 2 gana el equipo visitante y 3 empatan. Se encuentra en la carpeta de redecir y sus imágenes están guardadas en img. 
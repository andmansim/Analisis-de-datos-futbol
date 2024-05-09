# Analisis-de-datos-futbol
https://github.com/andmansim/Analisis-de-datos-futbol.git

csv datos_fut
columnas
club, country, part(num de participaciones), pld(partidos jugados), w(partidos gandos), d(partidos empatados), l(partidos perdidos), f(goles a favor), a(goles encontra), pts(puntos), gd(diferencia de goles), titles(veces ganada champions), porganarchampions(% ganar champions), porganarpartido (% ganar partido), porperderpartido(% perder partido), poremppartio(% empatar partido), porcapacidad_ofensiva, porcapacidad_defensiva, pordiferencia_ofen_defend, categoria(0 --> ataque, 1 --> defensa, 2 --> neutro)

Pasos:
1. Limpiamos los datos a examinar, los analizamos (ver si hay que quitar, añadir columnas o filas, transformar alguna variable categórica en numérica, etc ) y los representamos para ver su relación y que información nos dan. 
2. Hacemos una regresión lineal o logística para ver la relación entre las variables independientes (X) y las dependientes (Y) o para clasificar las distintas opciones o resultados. 
3. Hacemos la clusterización para ver patrones o tendencias en nuestros datos. 
4. Series temporales, es para ver la evolución de algunas variables a lo largo del tiempo en algunos grupos (clusters) identificados. 
5. Entrenamiento redes neuronales profundas, es un proceso de aprendizaje automático compuesto por múltiples capas de neuronas. Consiste en ajustar pesos y sesgos de las conexiones entre las neuronas para que el modelo pueda aprender a realizar tareas específicas.  
6. Entrenamiento redes neuronales convolucionales, es un proceso de aprendizaje automático que se utiliza para aplicar filtros en regiones locales y detectar texturas, patrones, formas, etc. en datos que tienen forma de cuadrícula como las imágenes y reconocimiento de patrones. Al igual que en la anterior tienen pesos y sesgos que se van ajustando según avanzamos en la red, donde las características más complejas se van fromando de las más sencillas de las capas anteriores. 

El 5. y 6. son practicamente igueles, 5. (DNN) donde hace operaciones metemáticas en los datos para pasarlos entre capas y 6. (CNN) se basa en patrones espaciales. 

7. Entrenamiento de una CNN mediente transferencia, aprovechamos el conocimiento adquirido por un modelo entrenado en una tarea similar. Por lo general se coge un modelo ya entrenado y se adapta a un nuevo uso más pequeño.

# Regresión logística
En esta parte presento dos posible enfoques a la hora de predecir el resultado de los partidos.
En regresion.py hace el análisis con la probabilidad de ganar, donde se realiza su probabilidad enfrenada en crear_csv. Por otro lado, en regresion1.py se realiza mediante la comparación de puntos, estos se asignan dependiendo de las comparaciones de los equipos en sus capacidades ofensivas y defensivas (crear_csv1.py).

Tras ver los resultados vemos que ambos métodos presentan los mismos resultados. 


Una regresión lineal o logística para ver la relación entre las variables independientes (X) y las dependientes (Y) o para clasificar las distintas opciones o resultados. 

Diferencia entre regresión logística y lineal:
Regresion logistica se utiliza cuando la variable de respuesta es binaria. Debe de ser un número entero.
Se utiliza para clasificar, no para predecir valores continuos. Ejemplo: predecir si un correo es spam o no.

Regresion lineal es para ver la relación entre dos variables.
Se utiliza para predecir valores continuos, ejemplo: predecir el precio de una casa.
'''
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

Todo esto sirve para ayudar a entrenar a la ia y examinar los datos que nos han dado. 
'''

import pandas as pd

#leemos el csv
df_equipos = pd.read_csv('datos_fut.csv', encoding='utf-8', sep=';')

#Examinamos los datos para ver si tenemos que añadir o eliminar columnas

#pensar cómo unir todos los datos de los partidos en uno solo. 
#¿lo tengo que unir al de los equipos? pq lo veo complicado
data = None

#Precedemos a hacer clusters con algoritmos centroides
#K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3) #inicializamos el modelo con 3 clusters
kmeans.fit(data) #aplicamos el modelo a los datos para encontrar esos clusters

#Mean Shift
from sklearn.cluster import MeanShift
meanshift = MeanShift() #lo mismo que con k-means
meanshift.fit(data)

#Mini Batch K-means
from sklearn.cluster import MiniBatchKMeans
minibatchkmeans = MiniBatchKMeans(n_clusters=3) #Es una versión más eficiente que con k-means
minibatchkmeans.fit(data) 

#Clusters basados en densidad
#DBSCAN

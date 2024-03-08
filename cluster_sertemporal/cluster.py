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
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import os


# Leemos el csv
df_equipos = pd.read_csv('csvs/datos_fut.csv', encoding='utf-8', sep=';')

# Seleccionamos solo las columnas numéricas
datos_num = df_equipos.select_dtypes(include=['float64', 'int64'])

# Función para aplicar un algoritmo de clustering y obtener métricas de desempeño
def aplicar_clustering(modelo, datos):
    modelo.fit(datos)
    #comprobamos si tiene las etiquetas, centroides y número de clusters
    #hasattr() es para saber si el objeto tiene un atributo
    
    if hasattr(modelo, 'labels_'): 
        labels = modelo.labels_
    elif hasattr(modelo, 'predict'):
        labels = modelo.predict(datos)
    else:
        labels = None

    if hasattr(modelo, 'cluster_centers_'):
        centroids = modelo.cluster_centers_
    else:
        centroids = None

    if hasattr(modelo, 'n_clusters'):
        n_clusters = modelo.n_clusters
    else:
        n_clusters = None

    # Mostrar los resultados
    print("\nModelo:", modelo)
    if n_clusters is not None:
        print("\nNúmero de clusters:", n_clusters)
    if centroids is not None:
        print("\nCentroides:", centroids)
    if labels is not None:
        print("\nEtiquetas:", labels)


#Procedemos a hacer clusters con algoritmos centroides
# Aplicar K-means
kmeans = KMeans(n_clusters=3) #inicializamos el modelo con 3 clusters
aplicar_clustering(kmeans, datos_num)

# Aplicar Mean Shift
meanshift = MeanShift() #inicializamos el modelo
aplicar_clustering(meanshift, datos_num)

# Aplicar Mini Batch K-means
minibatchkmeans = MiniBatchKMeans(n_clusters=3) #inicializamos el modelo con 3 clusters
aplicar_clustering(minibatchkmeans, datos_num)

#Clusters basados en densidad
# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
#eps es la distancia máxima entre dos muestras para que una sea considerada en el vecindario de la otra
#min_samples es el número mínimo de muestras en un vecindario para que una muestra sea considerada como un
#punto central
#un vecindario es una región de espacio que rodea un punto de datos
aplicar_clustering(dbscan, datos_num)

# Aplicar OPTICS
optics = OPTICS() #similar a DBSCAN pero no necesita los parámetros eps y min_samples
aplicar_clustering(optics, datos_num)


#Clusters basados en distribución
# Aplicar GMM
gmm = GaussianMixture(n_components=3)
aplicar_clustering(gmm, datos_num)

#Clusters jerárquicos
# Aplicar Agglomerative Clustering
agglomerative = AgglomerativeClustering(n_clusters=3)
aplicar_clustering(agglomerative, datos_num)



'''
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
'''
#dividimos los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Establecemos la X(representa el tiempo), y(os datos que cambian respecto a él)

# Establecer X (representa el tiempo) e y (los datos que cambian respecto a él)
X = df_equipos['W'] #Sale error pq aquí debe ir una fecha de verdad --> no es compatible con el modelo arima
y = df_equipos['porganarpartido']  

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo ARIMA
arima_model = ARIMA(y_train, order=(5, 1, 0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=len(y_test))

# Modelo Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train.to_frame(), y_train)

# Predicciones Random Forest
rf_forecast = rf_model.predict(X_test.to_frame())

# Calcular el error cuadrático medio
arima_error = mean_squared_error(y_test, arima_forecast)
rf_error = mean_squared_error(y_test, rf_forecast)

print("ARIMA MSE:", arima_error)
print("Random Forest MSE:", rf_error)

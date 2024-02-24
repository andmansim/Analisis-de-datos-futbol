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
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5) 
#eps es la distancia máxima entre dos muestras para que una sea considerada en el vecindario de la otra
#min_samples es el número mínimo de muestras en un vecindario para que una muestra sea considerada como un
#punto central
#un vecindario es una región de espacio que rodea un punto de datos
dbscan.fit(data)

#OPTICS
from sklearn.cluster import OPTICS
optics = OPTICS() #similar a DBSCAN pero no necesita los parámetros eps y min_samples
optics.fit(data)

#Clusters basados en distribución
#GMM
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3) #inicializamos el modelo con 3 clusters
gmm.fit(data) #aplicamos el modelo a los datos para encontrar esos clusters

#Clusters jerárquicos
#Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
agglomerative = AgglomerativeClustering(n_clusters=3) #inicializamos el modelo con 3 clusters
agglomerative.fit(data) #aplicamos el modelo a los datos para encontrar esos clusters



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

#Establecemos la X, y
X = None
y = None

train_data = None #es el csv con los datos de entrenamiento

#dividimos los datos en entrenamiento (80) y prueba(20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Modelo ARIMA
arima_model = ARIMA(train_data['value'], order=(5,1,0)) #modelo ARIMA con p=5, d=1, q=0. 
#p es el número de términos autorregresivos
#d es el número de diferencias no estacionales
#q es el número de términos de media móvil

arima_result = arima_model.fit()

# Predicciones ARIMA
arima_forecast = arima_result.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, typ='levels')
#start es el índice de la primera predicción
#end es el índice de la última predicción
#typ='levels' es para indicar que queremos las predicciones originales y no las diferencias


# Modelo Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(train_data[['feature1', 'feature2']], train_data['value'])
#feature1 y feature2 son las características que se usan para predecir el valor
#value es el valor que se quiere predecir


# Predicciones Random Forest
rf_forecast = rf_model.predict(test_data[['feature1', 'feature2']])


# Calcular error cuadrático medio, comparar los errores de los dos modelos
arima_error = mean_squared_error(test_data['value'], arima_forecast)
rf_error = mean_squared_error(test_data['value'], rf_forecast)

print("ARIMA MSE:", arima_error)
print("Random Forest MSE:", rf_error)
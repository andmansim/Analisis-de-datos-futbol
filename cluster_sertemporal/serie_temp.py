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
from cluster import *

#Establecemos la X(representa el tiempo), y(os datos que cambian respecto a él)

# Establecer X (representa el tiempo) e y (los datos que cambian respecto a él)
X = df_equipos['partganad'] #Sale error pq aquí debe ir una fecha de verdad --> no es compatible con el modelo arima
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

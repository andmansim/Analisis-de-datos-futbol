
#Regresión logística
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

#leemos el csv
df_partidos = pd.read_csv('partidos_fut.csv', encoding='utf-8', sep=';')

#separamos las variables independientes y dependientes
X = df_partidos[['prob_ganar_local', 'porb_empate', 'prob_ganar_visitante']]
y = df_partidos['resultado']

#Comparamos las probabilidades
df_partidos.loc[df_partidos['prob_ganar_local'] > df_partidos['prob_ganar_visitante'], 'resultado'] = 'local'
df_partidos.loc[df_partidos['prob_ganar_local'] < df_partidos['prob_ganar_visitante'], 'resultado'] = 'visitante'
df_partidos.loc[df_partidos['prob_ganar_local'] == df_partidos['prob_ganar_visitante'], 'resultado'] = 'empate'


#Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creamos el modelo
modelo = LogisticRegression()
#Entrenamos el modelo
modelo.fit(X_train, y_train)

#Hacemos las predicciones
y_pred = modelo.predict(X_test)

#Calculamos la precisión
print('Precisión del modelo:', accuracy_score(y_test, y_pred))

#Matriz de confusión
print(confusion_matrix(y_test, y_pred))

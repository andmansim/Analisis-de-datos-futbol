
#Regresión logística
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#leemos el csv
df = pd.read_csv('datos_fut.csv', encoding='utf-8', sep=';')

#separamos las variables independientes y dependientes
X = df[['porganarpartido', 'poremppartido', 'porperderpartido']]
y = df['Club']

#Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creamos el modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

#Hacemos las predicciones
y_pred = modelo.predict(X_test)

#Calculamos la precisión
print('Precisión del modelo:', accuracy_score(y_test, y_pred))

#Matriz de confusión
print(confusion_matrix(y_test, y_pred))

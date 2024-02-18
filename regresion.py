
#Regresión logística
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from crear_csv import actualizar_probabilidades

def regresion (X, y, df_partidos):
    
    #separamos las variables independientes y dependientes
    X = df_partidos[['prob_ganar_local', 'prob_empate', 'prob_ganar_visitante']]
    y = df_partidos['resultado']
    
    #Comparamos las probabilidades, para poder entrenar el modelo en función de estas
    #No pueden ser str tiene que ser int para que se puedan comparar y entrenarlo
    df_partidos.loc[df_partidos['prob_ganar_local'] > df_partidos['prob_ganar_visitante'], 'resultado'] = 1 #local
    df_partidos.loc[df_partidos['prob_ganar_local'] < df_partidos['prob_ganar_visitante'], 'resultado'] = 2 #visitante
    df_partidos.loc[df_partidos['prob_ganar_local'] == df_partidos['prob_ganar_visitante'], 'resultado'] = 3 #empate

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

    #Mostramos los resultados de cada partido que hemos obtenido de las predicciones
    predicciones = modelo.predict(X)
    for i, fila in df_partidos.iterrows():
        if predicciones[i] == 1:
            resultado = 'Local'
        elif predicciones[i] == 2:
            resultado = 'Visitante'
        else:
            resultado = 'Empate'
            
        print('\n' + fila['local'], 'vs', fila['visitante'], 'Resultado:', resultado)
    return df_partidos

#leemos el csv
df_partidos = pd.read_csv('partidos_fut.csv', encoding='utf-8', sep=';')

#llamamos a la función regresión
df_partidos = regresion(df_partidos)

#actualizamos los datos del csv con los resultados obtenidos
df_partidos.to_csv('partidos_fut.csv', index=False, sep=';')
print('Datos actualizados')

#en función de los resultados creamos un nuevo dataframe con los siguientes partidos
df_partidos_ronda2 = pd.DataFrame(columns=['local', 'visitante', 'prob_ganar_local', 'prob_empate', 'prob_ganar_visitante', 'resultado'])
df_partidos_ronda2['local']= ['Paris Saint-Germain', 'FC Barcelona', 'Club Atlético de Madrid', 'FC Bayern München', 'Arsenal FC', 'Manchester City FC', 'Borussia Dortmund', 'Real Madrid CF']
df_partidos_ronda2['visitante'] = ['Arsenal FC', 'Manchester City FC', 'Borussia Dortmund', 'Real Madrid CF', 'Paris Saint-Germain', 'FC Barcelona', 'Club Atlético de Madrid', 'FC Bayern München']
df_partidos_ronda2 = actualizar_probabilidades(df_partidos_ronda2)
print(df_partidos_ronda2.head())

df_partidos_ronda2 = regresion(df_partidos_ronda2)
df_partidos_ronda2.to_csv('partidos_fut_ronda2.csv', index=False, sep=';')
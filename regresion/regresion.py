'''
Regresion logistica se utiliza cuando la variable de respuesta es binaria. Debe de ser un número entero.
Se utiliza para clasificar, no para predecir valores continuos. Ejemplo: predecir si un correo es spam o no.

Regresion lineal es para ver la relación entre dos variables.
Se utiliza para predecir valores continuos, ejemplo: predecir el precio de una casa.
'''
#Regresión logística
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from crear_csv import actualizar_probabilidades

def regresion ( df_partidos):
    
    #Comparamos las probabilidades, para poder entrenar el modelo en función de estas
    #No pueden ser str tiene que ser int para que se puedan comparar y entrenarlo
    df_partidos['resultado'] = int(1) #local
    df_partidos.loc[df_partidos['prob_ganar_local'] < df_partidos['prob_ganar_visitante'], 'resultado'] = int(2) #visitante
    df_partidos.loc[df_partidos['prob_ganar_local'] == df_partidos['prob_ganar_visitante'], 'resultado'] = int(3) #empate
    
    print(df_partidos['resultado'])
    
    #separamos las variables independientes y dependientes
    X = df_partidos[['prob_ganar_local', 'prob_empate', 'prob_ganar_visitante']]
    y = df_partidos['resultado']
    
    #Dividimos los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, )
    print(X_train, X_test, y_train, y_test)
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
df_partido = pd.read_csv('csvs/partidos_fut_regresion.csv', encoding='utf-8', sep=';')

print('Ronda 1')
#llamamos a la función regresión
df_partido = regresion(df_partido)
#actualizamos los datos del csv con los resultados obtenidos
df_partido.to_csv('csvs/partidos_fut_ronda1.csv', index=False, sep=';')

#Ronda 2
#en función de los resultados creamos un nuevo dataframe con los siguientes partidos
df_partidos_ronda2 = pd.DataFrame(columns=['local', 'visitante', 'prob_ganar_local', 'prob_empate', 'prob_ganar_visitante', 'resultado'])
df_partidos_ronda2['local'] = ['Manchester City FC', 'FC Bayern München', 'Paris Saint-Germain', 'Club Atlético de Madrid', 'Real Madrid CF', 'Arsenal FC', 'FC Barcelona', 'Borussia Dortmund']
df_partidos_ronda2['visitante'] = ['Real Madrid CF', 'Arsenal FC', 'FC Barcelona', 'Borussia Dortmund', 'Manchester City FC', 'FC Bayern München', 'Paris Saint-Germain', 'Club Atlético de Madrid']
df_partidos_ronda2 = actualizar_probabilidades(df_partidos_ronda2)

print('Ronda 2')
df_partidos_ronda2 = regresion(df_partidos_ronda2)
df_partidos_ronda2.to_csv('csvs/partidos_fut_ronda2.csv', index=False, sep=';')


#Ronda 3
df_partidos_ronda3 = pd.DataFrame(columns=['local', 'visitante', 'prob_ganar_local', 'prob_empate', 'prob_ganar_visitante', 'resultado'])
df_partidos_ronda3['local']= ['Paris Saint-Germain','Real Madrid CF', 'Borussia Dortmund', 'FC Bayern München']
df_partidos_ronda3['visitante'] = ['Borussia Dortmund', 'FC Bayern München','Paris Saint-Germain',  'Real Madrid CF' ]
df_partidos_ronda3 = actualizar_probabilidades(df_partidos_ronda3)

print('Ronda 3')
df_partidos_ronda3 = regresion(df_partidos_ronda3)
df_partidos_ronda3.to_csv('csvs/partidos_fut_ronda3.csv', index=False, sep=';')

#Ronda 4
df_partidos_ronda4 = pd.DataFrame(columns=['local', 'visitante', 'prob_ganar_local', 'prob_empate', 'prob_ganar_visitante', 'resultado'])
df_partidos_ronda4['local']= ['Real Madrid CF', 'Borussia Dortmund']
df_partidos_ronda4['visitante'] = ['Borussia Dortmund' , 'Real Madrid CF']
df_partidos_ronda4 = actualizar_probabilidades(df_partidos_ronda4)

print('Ronda 4')
#cogemos los dos % de ganar de cada equipo y los comparamos
for i in range(len(df_partidos_ronda4)):
    if df_partidos_ronda4['prob_ganar_local'][i] < df_partidos_ronda4['prob_ganar_visitante'][i]:
        print('Gana el visitante', df_partidos_ronda4['visitante'][1])
        df_partidos_ronda4['resultado'][i] = 2
    else:
        print('Gana el local', df_partidos_ronda4['local'][0])
        df_partidos_ronda4['resultado'][i] = 1

df_partidos_ronda4.to_csv('csvs/partidos_fut_ronda4.csv', index=False, sep=';')

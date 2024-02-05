#Analizamos los datos de los csv goles.csv y winner.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Leemos los csv
df_goles = pd.read_csv('goles.csv', encoding='utf-8', delimiter=',')
df_ganador = pd.read_csv('winner.csv', encoding='utf-8', delimiter=',')

#Vemos la información de cada DataFrame
print('\n'+'Información de los DataFrames:')
print('\n'+'Goles:')
print(df_goles.info())
print(df_goles.columns)
print('\n'+'Ganador:')
print(df_ganador.info())
print(df_ganador.columns)
#No hay ningún valor nulo

def representar_barras(datos, datos2):
    plt.bar(datos, datos2)
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)
    plt.show()

print('\n'+ 'Representaciones gráficas:')
#Agrupamos por team(Equipo que metió gol en dicho minuto) y hacemos la media del minuto
df_goles_media = df_goles.groupby('team').mean()

#Representamos los 15 equipos con mayor media de goles
df_goles_media = df_goles_media.sort_values(by='minute', ascending=False)
df_goles_media = df_goles_media.head(15)
print(df_goles_media)
representar_barras(df_goles_media.index, df_goles_media['minute'])


#Agrupamos y contamos las veces que han jugado en casa y fuera
df_ganadorhome= df_ganador['home_team'].value_counts()
df_ganadoraway = df_ganador['away_team'].value_counts()

#Representamos los 15 equipos con mayor media de goles en casa y fuera
df_ganadorhome = df_ganadorhome.sort_values(ascending=False)
df_ganadorhome = df_ganadorhome.head(15)
representar_barras(df_ganadorhome.index, df_ganadorhome)

df_ganadoraway = df_ganadoraway.sort_values(ascending=False)
df_ganadoraway = df_ganadoraway.head(15)
representar_barras(df_ganadoraway.index, df_ganadoraway)



#Hacemos un análisis de los datos

print('\n'+ 'Análisis de los datos:')
#Contamos las veces que ha ganado cada equipo
print('\n'+ 'Las veces que ha ganado cada equipo:')
df_ganador = df_ganador['winner'].value_counts()
print(df_ganador)

#Contamos cuantas veces ha marcado gol el jugador 
print('\n'+ 'Ranking de goles de cada jugador:')
df_golesjugadores = df_goles['scorer'].value_counts()
print(df_golesjugadores)

#Media de goles por minuto
media_goles = df_goles['minute'].mean()
print('\n'+'Media de goles por minuto:', media_goles)

#Varianza y desviación estándar de los goles
varianza_goles = df_goles['minute'].var()
desviacion_goles = df_goles['minute'].std()
print('\n'+'Varianza de los goles:', varianza_goles)
print('\n'+'Desviación estándar de los goles:', desviacion_goles)




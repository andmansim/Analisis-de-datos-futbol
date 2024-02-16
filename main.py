import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#leemos el csv
df = pd.read_csv('datos_fut.csv', encoding='utf-8', sep=';')
print(df.head())  

#vemos las columnas y los datos
print(df.columns)
print(df.info())  

#Representación de los datos
def grafi_barras(dato1, dato2, titulo, x, y):
    plt.figure(figsize=(10,5))
    plt.bar(dato1, dato2)
    plt.title(titulo)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

df_ordenado = df.sort_values(by='Titles', ascending=False)
df_ordenado = df_ordenado.head(10)
grafi_barras(df_ordenado['Club'], df_ordenado['Titles'], 'Número de veces que ha ganado la champions cada equipo', 'Equipo', 'Número de veces')


#Gráfico de barras, mostramos cuantos equipos hay por país
equipo_pais = df['Country'].value_counts()
grafi_barras(equipo_pais.index, equipo_pais.values, 'Equipos por país', 'País', 'Número de equipos')


 

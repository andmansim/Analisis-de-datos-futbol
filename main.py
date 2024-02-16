import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#leemos el csv
df = pd.read_csv('datos_fut.csv', encoding='utf-8', sep=';')
print(df.head())  

#vemos las columnas y los datos
print(df.columns)
print(df.info())  

#Remplazamos las comas por puntos
df = df.replace(',', '.', regex=True)
print(df.head())
#convertimos las columnas a float
for i in df.columns:
    if 'por' in i:
        df[i] = df[i].astype(float)
print(df.info())


#Representación de los datos
def grafi_barras(dato1, dato2, titulo, x, y):
    plt.figure(figsize=(12,5))
    plt.bar(dato1, dato2)
    plt.title(titulo)
    plt.xlabel(x)
    plt.xticks(rotation=45)
    plt.ylabel(y)
    plt.show()

def grafi_baras_colores(dato1, titulo, x, y):
    dato1.plot(kind='bar', color=['green', 'blue', 'red'], stacked=False, figsize=(12,5))
    plt.title(titulo)
    plt.xlabel(x)
    plt.xticks(rotation=45)
    plt.ylabel(y)
    plt.show()
    
#Gráficos de barras, 10 que más han ganado la champions
df_ordenado = df.sort_values(by='Titles', ascending=False)
df_ordenado = df_ordenado.head(10)
grafi_barras(df_ordenado['Club'], df_ordenado['Titles'], 'Número de veces que ha ganado la champions cada equipo', 'Equipo', 'Número de veces')


#Gráfico de barras, equipos hay por país
equipo_pais = df['Country'].value_counts()
grafi_barras(equipo_pais.index, equipo_pais.values, 'Equipos por país', 'País', 'Número de equipos')

#Gráfico de barras mostrando los el % de ataque, defensa de 10 mejores equipos
df_por = df.set_index('Club')[["porganarpartido", 'poremppartido', 'porperderpartido']]
grafi_baras_colores(df_por.head(10), 'Porcentaje de ataque, defensa y empate de los 10 mejores equipos', 'Equipo', 'Porcentaje')


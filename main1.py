#Analizamos los datos de los csv equipo.csv y partidos.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


#leemos los datos
def leer_csv(ruta, indice):
    #Función encargada en leer y asignar los nombres de las columnas correspondientes
    with open (ruta, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        df = pd.DataFrame(reader, columns = indice)
    return df

#Leemos cada csv y creamos los DataFrames correspondientes
df_equipo = leer_csv('equipo.csv', ['nombre', 'img', 'pais', 'id'])
df_partidos = leer_csv('partidos.csv', ['dia', 'equipo1', 'equipo2', 'numero1', 'numero2', 'numero3'])

#Vemos la información de cada DataFrame
print('\n'+'Información de los DataFrames:')
print('\n'+'Equipo:')
print(df_equipo.info())
print('\n'+'Partidos:')
print(df_partidos.info())
#No hay ningún valor nulo

#Nos damos cuenta que las 3 últimas columnas no sirven para nada, entonces vamos a quitar 1 y las otras las cambiaremos 
# por posible goles que haya metido cada equipo
df_partidos = df_partidos.drop(columns=['numero1', 'numero2', 'numero3'])
df_partidos['goles1'] = np.random.randint(0, 5, df_partidos.shape[0])
df_partidos['goles2'] = np.random.randint(0, 5, df_partidos.shape[0])
print(df_partidos.head())


#Vamos a hacer un análisis de los datos

#Juntamos los dos DataFrames
df = df_partidos.merge(df_equipo, left_on='equipo1', right_on='id')
print('\n'+'DataFrame final:')
print(df)

#Agrupamos según el id y sumamos los goles de cada equipo por separado
goles_totales1 = df.groupby('id')['goles1'].sum().reset_index()
goles_totales2 = df.groupby('id')['goles2'].sum().reset_index()

#Juntamos los dos dataframe en 1 para obtener los goles totales de todos los equipos
goles_totales = pd.merge(goles_totales1, goles_totales2, on='id')
goles_totales['goles'] = goles_totales['goles1'] + goles_totales['goles2']
print('\n' + 'goles totales de cada equipo')
print(goles_totales)

#Calculamos la media de goles por id
media_goles = goles_totales.mean()
print('\n'+'Media de goles por id:', media_goles)

#Calculamos el id con más y menos goles
print('\n'+'Id con más goles:', goles_totales[goles_totales['goles'] == goles_totales['goles'].max()])
print('\n'+'Id con menos goles:', goles_totales[goles_totales['goles'] == goles_totales['goles'].min()])

#Varianza y desviación estándar de los goles
varianza_goles = goles_totales['goles'].var()
desviacion_goles = goles_totales['goles'].std()
print('\n'+'Varianza de los goles:', varianza_goles)
print('\n'+'Desviación estándar de los goles:', desviacion_goles)


# Representación de los equipos por país 
plt.bar(df_equipo['pais'].value_counts().index, df_equipo['pais'].value_counts())
plt.xticks(rotation=45, ha='right')  # Ajusta la rotación y alineación horizontal de los nombres
plt.xlabel('País')
plt.ylabel('Número de Equipos')
plt.title('Número de Equipos por País')
plt.tight_layout()
plt.show()

#Representación de los goles por equipo
plt.bar(goles_totales['id'], goles_totales['goles'])
plt.xlabel('Id')
plt.ylabel('Número de Goles')
plt.title('Número de Goles por Id')
plt.tight_layout()
plt.show()

#Representación de los goles por país
df = pd.merge(df_equipo, goles_totales, on = 'id')
plt.bar(df['pais'].value_counts().index, df['pais'].value_counts())
plt.xticks(rotation=45, ha='right')  # Ajusta la rotación y alineación horizontal de los nombres
plt.xlabel('País')
plt.ylabel('Número de Goles')
plt.title('Número de Goles por País')
plt.tight_layout()
plt.show()



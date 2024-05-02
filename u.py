import pandas as pd

'''df = pd.read_csv('csvs/datos_fut.csv', encoding='utf-8', sep=';')
print(df.info())


archivo_salida = 'datos_con_comas.csv'



# Cambiar el separador decimal de punto a coma en todas las columnas numéricas
datos = df.applymap(lambda x: str(x).replace('.', ','))

# Guardar los datos modificados en el archivo CSV de salida
datos.to_csv(archivo_salida, index=False, sep=';')

print("Se ha modificado el separador decimal de punto a coma en el archivo CSV utilizando Pandas.")
print(datos.info())
'''
df = pd.read_csv('csvs/partidos22_23.csv', encoding='utf-8', sep=';')
#remplazamos , por . de todas las columnas
for columna in df.select_dtypes(include=['object']):
    if df[columna].str.contains(',').any():
        df[columna] = df[columna].str.replace(',', '.')
        df[columna] = df[columna].astype(float)

print(df.info())
df.to_csv('csvs/partidos22_23.csv', sep=';', index=False, encoding='utf-8')
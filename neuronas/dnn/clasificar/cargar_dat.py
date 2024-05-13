import pandas as pd
import numpy as np
import torch
from neurona_profunda import RedNeuronal
import os
from sklearn.model_selection import train_test_split

'''#cargamos el modelo
#cogemos los datos a clasificar
df = pd.read_csv('csvs/datos_fut.csv', delimiter=';', encoding='utf-8')
df = df.drop(['club', 'pais'], axis=1)
df_equipos = pd.read_csv('csvs/datos_fut.csv', delimiter=';', encoding='utf-8')
x_nuevos = df_equipos.drop(['porganarpartido', 'porperderpartido', 'poremppartido'], axis=1)
#Dejamos solo las 5 primeras filas
x_nuevos = x_nuevos.head(5)

hl = 10
x_train = 15
#cargamos el modelo
model = RedNeuronal()
modelo_ruta = os.path.join(os.path.dirname(__file__), 'modelo_red_neuronal_prof_uefa.pth')
model.load_state_dict(torch.load(modelo_ruta))
model.eval()

#hacemos las predicciones
x = torch.Tensor(x_nuevos.values).float()
_, predicted = torch.max(model(x).data, 1)
print('Predicciones:\n',predicted)'''

df_equipos = pd.read_csv('csvs/datos_fut_clasificados.csv', delimiter=',', encoding='utf-8')
df_equipos = df_equipos.drop(['club', 'pais'], axis=1)

#Separamos los datos en train y test
x = df_equipos.drop(['porganarpartido', 'porperderpartido', 'poremppartido'], axis=1)
y = df_equipos['categoria']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape[1])

df = pd.read_csv('csvs/datos_fut.csv', delimiter=';', encoding='utf-8')
x_nuevos = df.drop(['club', 'pais'], axis=1)
print(x_nuevos.shape[1])
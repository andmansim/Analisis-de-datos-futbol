'''
Durante el entrenamiento se le proporciona un conjunto de datos de entrada y salida esperada, 
y el algoritmo ajusta los pesos de las conexiones entre las neuronas para minimizar el error 
entre la salida esperada y la salida real.
'''
import torch
import torch.nn as nn
import torch.utils.data as td
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


#Aquí vamos a clasificar los equipos en tres categorías, ataque, defensa o neutro
#dependiendo de ello les asociaremos un número 0, 1 o 2

#leemos el archivo
df_equipos = pd.read_csv('csvs/datos_fut.csv', delimiter=';', encoding='utf-8')

#clasificamos los equipos
def clasificar_equipos(row):
    if row['porgfav/gtot'] > 50:
        return 0 #ataque
    elif row['porgfav/gtot'] < 50:
        return 1 #defensa
    else:
        return 2 #neutro

#creamos una nueva columna en el dataframe
df_equipos['categoria'] = df_equipos.apply(clasificar_equipos, axis=1)

#guardamos los datos en un nuevo archivo
df_equipos.to_csv('csvs/datos_fut_clasificados.csv', index=False)

#Quitamos las columnas que no numéricas
df_equipos = df_equipos.drop(['Club', 'Country'], axis=1)

#creamos semilla
torch.manual_seed(0)
print('Se han importado las librerías, listo para usar\n', torch.__version__)

#Separamos los datos en train y test
x = df_equipos.drop(['porganarpartido', 'porperderpartido', 'poremppartido'], axis=1)
y = df_equipos['categoria']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('Datos separados en train y test\n')
print('x_train:', x_train.shape
      , 'x_test:', x_test.shape
      , 'y_train:', y_train.shape
      , 'y_test:', y_test.shape)
#preparamos los datos 
#creamos un dataset con los datos de train
train_x = torch.Tensor(x_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=20, shuffle=False, num_workers=1)








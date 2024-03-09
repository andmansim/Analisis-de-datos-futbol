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
    if row['porgfav/gtot'] > 60:
        return 0
    elif row['porgfav/gtot'] < 40:
        return 1
    else:
        return 2



#importamos librerias
torch.manual_seed(0)
print('Se han importado las librerías, listo para usar\n', torch.__version__)

#preparamos los datos 
#creamos un dataset con los datos de train
train_x = torch.Tensor(x_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=20, shuffle=False, num_workers=1)








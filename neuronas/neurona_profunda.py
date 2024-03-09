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

#preparamos los datos para torch
#creamos un dataset con los datos de train
train_x = torch.Tensor(x_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=20, shuffle=False, num_workers=1)

#definimos neuronas
#definimos el número de nodos en cada capa oculta
hl = 10

#definimos la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self):
        super(RedNeuronal, self).__init__()#se inicializa la clase padre
        
        #se define la capa de entrada que toma como entrada la cantidad de nodos (los valores de entrada)
        #y produce una salida de hl nodos
        self.fc1 = nn.Linear(x_train.shape[1], hl) 
        
        #se define la segunda capa oculta con hl nodos y produce una salida de hl nodos
        self.fc2 = nn.Linear(hl, hl)
        
        #se define la capa de salida con hl nodos y produce una salida de 3 nodos
        self.fc3 = nn.Linear(hl, x_train.shape[1])

    def forward(self, x):
        #indica cómo van a ser procesados los datos de entrada
        #en cada x pasamos los datos por la capa oculta y aplicamos la función de activación
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        #nos devuelve la salida final de los datos
        return x

#creamos una instancia de la red neuronal
model = RedNeuronal()
print('Red Neuronal creada\n', model)

#Entrenamos la red neuronal

def train(model, data_loader, optimizador):
    #indicamos que la red está en modo de entrenamiento
    model.train()
    
    #inicializamos el error
    train_loss = 0
    
    for batch, tensor in enumerate(data_loader):
        #iteramos a través de los datos para obtener lotes de datos y 
        #realizar un seguimiento del número de lotes con batch
        
        #obtenemos los datos y las etiquetas del lote actual
        data, target = tensor
        
        #reiniciamos el optimizador a cero para que no se 
        #acumulen los gradientes
        optimizador.zero_grad()
        
        #pasamos los datos por la red
        output = model(data)
        
        #calculamos el error (pedida de los datos reales con los 
        #datos predichos)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        #acumulamos el error
        train_loss += loss.item()
        
        #se realiza la retropropagación (pase hacia atrás) 
        #para ajustar los pesos
        loss.backward()
        
        #actualizamos los parámetros de la red
        optimizador.step()
        
    #devolvemos la media del error
    media_error = train_loss / (batch + 1)
    print('Error medio:', media_error)
    return media_error









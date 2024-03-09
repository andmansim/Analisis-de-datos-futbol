'''
Durante el entrenamiento se le proporciona un conjunto de datos de entrada y salida esperada, 
y el algoritmo ajusta los pesos de las conexiones entre las neuronas para minimizar el error 
entre la salida esperada y la salida real.
'''
import torch
import torch.nn as nn
import torch.utils.data as td


#importamos librerias
torch.manual_seed(0)
print('Se han importado las librerías, listo para usar\n', torch.__version__)

#preparamos los datos 
#creamos un dataset con los datos de train
train_x = torch.Tensor(x_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=20, shuffle=False, num_workers=1)








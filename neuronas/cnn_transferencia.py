'''
Las CNN que aprenden por transferenccia son más rápidas y precisas que las CNN que aprenden desde cero, 
debido a que las primeras ya tienen un conocimiento previo de los datos.
Vamos a adaptar una CNN preentrenada en imagenes para el csv de futbol
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import torch.utils.data as td
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torchvision.models as models

#preparamos la base del modelo
model = torchvision.models.resnet18(pretrained=True)



# Cargar los datos desde el CSV
df_equipos = pd.read_csv('csvs/datos_fut_clasificados.csv', encoding='utf-8', delimiter=',')

# Eliminar las variables categóricas
df_equipos = df_equipos.drop(['Club', 'Country'], axis=1)


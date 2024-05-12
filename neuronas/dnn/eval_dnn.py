import os
import torch
import torch.nn as nn
import torch.utils.data as td
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from dnn import RedNeuronal

modelo_ruta = 'neuronas/dnn/modelo_dnn_uefa.pth'
#cargamos el modelo
model = RedNeuronal()
model.load_state_dict(torch.load(modelo_ruta))
model.eval()
x_nuevos = None
x = torch.Tensor(x_nuevos.values).float()
_, predicted = torch.max(model(x).data, 1)
print('Predicciones:\n',predicted.items())

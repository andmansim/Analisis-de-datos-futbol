import pandas as pd
import torch
from entrenar_dnn import RedNeuronal
import os

'''
Mandamos datos a la red neuronal para clasificar los jugadores de futbol en base a su rendimiento
'''

#cogemos los datos a clasificar
df = pd.read_csv('csvs/datos_fut.csv', delimiter=';', encoding='utf-8')
x_nuevos = df.drop(['club', 'pais'], axis=1)
#Dejamos solo las 5 primeras filas
x_nuevos = x_nuevos.head(5)

hl = 10
entrada = x_nuevos.shape[1]

#cargamos el modelo
model = RedNeuronal(entrada, hl)
modelo_ruta = os.path.join(os.path.dirname(__file__), 'modelo_red_neuronal_prof_uefa.pth')
model.load_state_dict(torch.load(modelo_ruta))
model.eval()

#hacemos las predicciones
x = torch.Tensor(x_nuevos.values).float()
_, predicted = torch.max(model(x).data, 1)
print('Predicciones:\n',predicted)
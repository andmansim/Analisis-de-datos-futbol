import pandas as pd
import os
import torch
from entrenar_dnn import RedNeuronal
from sklearn.model_selection import train_test_split
'''
Mandamos datos a la red neuronal para predecir los resultados de los partidos de la uefa champions league
'''

#leemos el csv de los datos a predecir
df_23_24 = pd.read_csv('csvs/partidos_fut_dnn_23_24.csv', delimiter=';', encoding='utf-8')
# Aplicar one-hot encoding al nombre del club
df_23_24 = pd.get_dummies(df_23_24, columns=['local', 'visitante'])
#Eliminamos la columna de resultado
df_23_24 = df_23_24.drop(columns=['resultado'])

#preparamos los datos para torch
hl = 10
tipo_resultados = [1,2,3] #1 gana local, 2 gana visitante y 3 empatan
features = ['porganarpartido_local','porganarpartido_visitante','porperderpartido_local', 'porperderpartido_visitante', 'porcapacidad_ofensiva_local','porcapacidad_ofensiva_visitante', 'porcapacidad_defensiva_local','porcapacidad_defensiva_visitante']
x_nuevos = df_23_24[features]

#creamos un modelo vacio
model = RedNeuronal(features, tipo_resultados, hl)
#cargamos el modelo guardado
modelo_ruta = os.path.join(os.path.dirname(__file__), 'modelo_dnn_uefa.pth')
model.load_state_dict(torch.load(modelo_ruta))
#evaluamos el modelo
model.eval()


#le pasamos los datos a la red
x = torch.Tensor(x_nuevos.values).float()
#obtenemos las predicciones
_, predicted = torch.max(model(x).data, 1)
print('Predicciones:\n',predicted)


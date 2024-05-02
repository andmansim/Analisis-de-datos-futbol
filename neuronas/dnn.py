'''
Durante el entrenamiento se le proporciona un conjunto de datos de entrada y salida esperada, 
y el algoritmo ajusta los pesos de las conexiones entre las neuronas para minimizar el error 
entre la salida esperada y la salida real.
'''
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


#Aquí vamos a clasificar los equipos en tres categorías, ataque, defensa o neutro
#dependiendo de ello les asociaremos un número 0, 1 o 2

#leemos el archivo
df_equipos = pd.read_csv('csvs/partidos22_23.csv', delimiter=',', encoding='utf-8')


#Quitamos las columnas que no numéricas
df_equipos = df_equipos.drop(['ronda'], axis=1)

#creamos semilla
torch.manual_seed(0)
print('Se han importado las librerías, listo para usar\n', torch.__version__)

# Aplicar one-hot encoding al nombre del club
df_equipos = pd.get_dummies(df_equipos, columns=['local', 'visitante'])


features = ['porganarpartido_local','porganarpartido_visitante' 'porperderpartido_local', 'porperderpartido_visitante', 'porcapacidad_ofensiva_local','porcapacidad_ofensiva_visitante', 'porcapacidad_defensiva_local','porcapacidad_defensiva_visitante']
#Separamos los datos en train y test
x = df_equipos[features]
y = df_equipos['resultado']

#preparamos los datos para torch
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('Datos separados en train y test\n')
for i in range(10):
    print(x_train.iloc[i], y_train.iloc[i])

print('TODO BIEN 1\n')
#preparamos los datos para torch
#creamos un dataset con los datos de train
train_x = torch.Tensor(x_train.values).float()
train_y = torch.Tensor(y_train.values).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=20, shuffle=False, num_workers=1)

#creamos un dataset con los datos de test
test_x = torch.Tensor(x_test.values).float()
test_y = torch.Tensor(y_test.values).long()
test_ds = td.TensorDataset(test_x, test_y)
test_loader = td.DataLoader(test_ds, batch_size=20, shuffle=False, num_workers=1)

print('TODO BIEN 2\n')

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

print('TODO BIEN 3\n')

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



#definimos los test
def test(model, data_loader):
    #indicamos que la red está en modo de evaluación
    #no hay retropropagación
    model.eval()
    
    #inicializamos el error
    test_loss = 0
    #inicializamos el número de predicciones correctas
    correct = 0
    
    with torch.no_grad():
        #Establecemos un bucle para evaluar los datos 
        #sin la necesidad de calcular los gradientes
        batch_count = 0
        
        for barch, tensor in enumerate(data_loader):
            #iteramos a través de los datos para obtener 
            #lotes de datos
            
            batch_count += 1
            
            #obtenemos los datos y las etiquetas del lote actual
            data, target = tensor
            
            #hacemos el pase hacia delante (feedforward)
            output = model(data)
            
            #calculamos el error acumulado
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            
            #calculamos el número de predicciones correctas
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()
    
    media_error = test_loss / batch_count
    print('Prueba: pérdida media: %f, precisión: %f ' % (test_loss, 100. * correct /len(data_loader.dataset)))
    return media_error

loss_criteria = nn.CrossEntropyLoss()
#Tasa de aprendizaje 
learning_rate = 0.001
#acrtualiza los pesos durente el entrenamiento para minimazar la función pérdida
optimizador = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizador.zero_grad()

if __name__ == '__main__':
    #epoch o épocas son cada pasada completa por el conjunto de datos de entrenamiento
    epoch_nums = []
    training_loss = [] 
    validation_loss = []

    #entrenamos la red
    epochs = 50
    for epoch in range(1, epochs + 1): #iteramos a través de las épocas
        print('Epoch:', epoch)
        train_loss = train(model, train_loader, optimizador)
        test_loss = test(model, train_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
    
    #graficamos el error
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    
    for param_tensor in model.state_dict():
        print(param_tensor, "\n", model.state_dict()[param_tensor].size(), '\n', model.state_dict()[param_tensor])
    
    #evaluamos el modelo
    model.eval()
    x1 = torch.Tensor(x_test.values).float()
    _, predicted = torch.max(model(x1).data, 1)
    
    #creamos la matriz de confusión
    matriz = confusion_matrix(y_test, predicted.numpy())
    plt.imshow(matriz, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(x_train.shape[1])
    plt.xticks(tick_marks, x_train.columns, rotation=45)
    plt.yticks(tick_marks, x_train.columns)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.show()
    
    #guardamos el modelo
    modelo_ruta = os.path.join(os.path.dirname(__file__), 'modelo_dnn_uefa.pth')
    torch.save(model.state_dict(), modelo_ruta)
    del model
    print('Modelo guardado en', modelo_ruta)
    
''' #cargamos el modelo
    model = RedNeuronal()
    model.load_state_dict(torch.load(modelo_ruta))
    model.eval()
    x_nuevos = None
    x = torch.Tensor(x_nuevos.values).float()
    _, predicted = torch.max(model(x).data, 1)
    print('Predicciones:\n',predicted.items())
    '''
    
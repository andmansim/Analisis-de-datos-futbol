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


#definimos la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self, features, tipo_resultados, hl):
        super(RedNeuronal, self).__init__()#se inicializa la clase padre
        
        #se define la capa de entrada que toma como entrada la cantidad de nodos (los valores de entrada)
        #y produce una salida de hl nodos
        self.fc1 = nn.Linear(len(features), hl) 
        
        #se define la segunda capa oculta con hl nodos y produce una salida de hl nodos
        self.fc2 = nn.Linear(hl, hl)
        
        #se define la capa de salida con hl nodos y produce una salida de 3 nodos
        self.fc3 = nn.Linear(hl, len(tipo_resultados))

    def forward(self, x):
        #indica cómo van a ser procesados los datos de entrada
        #en cada x pasamos los datos por la capa oculta y aplicamos la función de activación
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        #nos devuelve la salida final de los datos
        return x



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
        loss = loss_criteria(output, target)
        
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
            test_loss += loss_criteria(output, target).item()
            
            #calculamos el número de predicciones correctas
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()
    
    media_error = test_loss / batch_count
    print('Perdida media:', media_error, 'Predicciones correctas:', correct, 'Total de predicciones:', len(data_loader.dataset))
    return media_error




if __name__ == '__main__':
    
    #Aquí vamos a clasificar los equipos en tres categorías, ataque, defensa o neutro
    #dependiendo de ello les asociaremos un número 0, 1 o 2

    #leemos el archivo
    df_equipos = pd.read_csv('csvs/partidos_fut_dnn.csv', delimiter=';', encoding='utf-8')
    
    
    # quitamos local y visitante
    df_equipos = df_equipos.drop(columns=['local', 'visitante'])
    

    #Separamos los datos en train y test
    tipo_resultados = [1,2,3] #1 gana local, 2 gana visitante y 3 empatan
    x = df_equipos.drop(columns=['resultado'])
    features = x.columns.tolist()
    y = df_equipos['resultado']
    
    
    x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.30, random_state=0)

    print('Datos separados en train y test\n')
    for i in range(10):
        print(x_train[i], y_train[i])

    #creamos semilla
    torch.manual_seed(0)
    print('Se han importado las librerías, listo para usar\n', torch.__version__)
    

#preparamos los datos para torch
    #creamos un dataset con los datos de train
    train_x = torch.Tensor(x_train).float()
    train_y = torch.Tensor(y_train).long()
    train_ds = td.TensorDataset(train_x, train_y)
    train_loader = td.DataLoader(train_ds, batch_size=20, shuffle=False, num_workers=1)

    #creamos un dataset con los datos de test
    test_x = torch.Tensor(x_test).float()
    test_y = torch.Tensor(y_test).long()
    test_ds = td.TensorDataset(test_x, test_y)
    test_loader = td.DataLoader(test_ds, batch_size=20, shuffle=False, num_workers=1)

    
#creamos una instancia de la red neuronal
    #definimos el número de nodos en cada capa oculta
    hl = 10
    model = RedNeuronal(features, tipo_resultados, hl)
    print('Red Neuronal creada\n', model)

    loss_criteria = nn.CrossEntropyLoss()
    #Tasa de aprendizaje 
    learning_rate = 0.0001
    #actualiza los pesos durante el entrenamiento para minimazar la función pérdida
    optimizador = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizador.zero_grad()

    #epoch o épocas son cada pasada completa por el conjunto de datos de entrenamiento
    epoch_nums = []
    training_loss = [] 
    validation_loss = []
    
#entrenamos la red
    epochs = 50
    for epoch in range(1, epochs + 1): #iteramos a través de las épocas
        print('Epoch:', epoch)
        train_loss = train(model, train_loader, optimizador)
        test_loss = test(model, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

#graficamos el error
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig('neuronas/dnn/predecir/img/grafica_error_dnn.png')
    plt.show()

    for param_tensor in model.state_dict():
        print(param_tensor, "\n", model.state_dict()[param_tensor].size(), '\n', model.state_dict()[param_tensor])

#evaluamos el modelo
    model.eval()
    x1 = torch.Tensor(x_test).float()
    _, predicted = torch.max(model(x1).data, 1)

#creamos la matriz de confusión
    matriz = confusion_matrix(y_test, predicted.numpy())
    plt.imshow(matriz, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(tipo_resultados))
    plt.xticks(tick_marks, tipo_resultados, rotation=45)
    plt.yticks(tick_marks, tipo_resultados)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.savefig('neuronas/dnn/predecir/img/confusion_matrix_dnn.png')
    plt.show()

#guardamos el modelo
    modelo_ruta = os.path.join(os.path.dirname(__file__), 'modelo_dnn_uefa.pth')
    torch.save(model.state_dict(), modelo_ruta)
    del model
    print('Modelo guardado en', modelo_ruta)
    

    



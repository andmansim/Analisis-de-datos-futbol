
'''
Vamos a entrenar una CNN(red neuronal convolucional) capaz de clasificar
imagenes de formas geométricas simples. 
En nuestro caso lo adaptamos para que clasifique los equipos en función de 3 categorías, 
defensa, ataque o neutro. 

'''

#Importamos las librerías
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as td
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os
import pandas as pd

#Cargamos los datos
df_equipos = pd.read_csv('csvs/datos_fut_clasificados.csv', encoding='utf-8', delimiter=',')

#Eliminamos las variables categóricas
df_equipos = df_equipos.drop(['club', 'pais'], axis=1)

#Dividimos los datos en x, y
X = df_equipos.drop(['porganarpartido', 'porperderpartido', 'poremppartido'], axis=1).values
y = df_equipos['categoria']

# Realizar cualquier preprocesamiento necesario y convertir los datos en tensores PyTorch
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Agregar una dimensión de canal
y = torch.tensor(y.values, dtype=torch.long)


#Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear conjuntos de datos y cargadores de datos
train_dataset = td.TensorDataset(X_train, y_train)
test_dataset = td.TensorDataset(X_test, y_test)

#creamos los cargadores de datos
batch_size = 64 #indica el num de muestras de cada lote
train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('Datos cargados\n')

#creamos la red neuronal
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        
        # Definir las capas de la red neuronal con sus entradas y las salidas esperadas
        self.fc1 = nn.Linear(input_size * 15, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Aplanar la entrada y pasarla a través de las capas de la red neuronal
        #Estas tienen la forma de (batch_size, input_size), que e slo que espera la red neuronal
        #batch_size es el número de muestras en el lote
        #input_size es el número de características en cada muestra
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
print('Red Neuronal creada\n')  



#entremaos el modelo
def train(model, loss_criteria, optimizer, train_loader):
    #activamos el modo de entrenamiento
    model.train()
    
    #inicializamos las variables
    train_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(train_loader, 0):
        #obtenemos las entradas y las etiquetas del lote actual
        inputs, labels = data
        #reiniciamos los gradientes 
        optimizer.zero_grad()
        #pasamos los datos del modelo, es decir, hacemos una predicción hacia adelante
        outputs = model(inputs)
        
        # Ajustar dimensiones de las etiquetas
        labels = labels.view(-1)
        labels = labels - 1
        
        #calculamos la pérdida
        loss = loss_criteria(outputs, labels)
        #hacemos la propagación hacia atrás para calcular los gradientes
        loss.backward()
        #actualizamos los pesos (parámetros) de la red neuronal
        optimizer.step()
        #sumamos la pérdida
        train_loss += loss.item()
        
        #calculamos el número de predicciones correctas
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        #calculamos el número total de muestras
        total += labels.size(0)
        
    #mostramos el promedio de la pérdida y la precisión
    print('Train set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
        train_loss / len(train_loader.dataset), 100 * correct / total))
    
    return train_loss / len(train_loader.dataset), correct / total
    

def test(model, loss_criteria, test_loader):
    #activamos el modo de evaluación
    model.eval()
    
    #inicializamos las variables
    test_loss = 0.0
    correct = 0
    total = 0
    
    #Desactivamos el cálculo de gradientes
    with torch.no_grad():
        #iteramos sobre los datos de prueba
        for data in test_loader:
            #obtenemos las entradas y las etiquetas del lote actual
            inputs, labels = data
            
            # Ajustar dimensiones de las etiquetas
            labels = labels.view(-1)            
            labels = labels - 1
            
            #pasamos los datos del modelo, es decir, hacemos una predicción hacia adelante
            outputs = model(inputs)
            #calculamos la pérdida
            loss = loss_criteria(outputs, labels)
            #sumamos la pérdida
            test_loss += loss.item()
            
            #calculamos el número de predicciones correctas e incrementamos el total
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    #mostramos el promedio de la pérdida y la precisión
    accuracy = 100 * correct / total
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
        test_loss / len(test_loader.dataset), accuracy))
    
    return test_loss / len(test_loader.dataset), accuracy

#Main
#definimos el tamaño de la entrada y el número de clases
input_size = X.shape[1] #número de columnas de X
num_classes = 3 #número de clases diferentes (el número de variables que los queremos clasificar)

#creamos el modelo
model = Net(input_size, num_classes)
#definimos la función de pérdida y el optimizador
loss_criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


if __name__ == '__main__':
    #inicializamos las listas para almacenar los resultados
    epoch_nums = []
    training_loss = []
    validation_loss = []

    #entrenamos el modelo
    epochs = 5
    
    #iteramos sobre el número de épocas y mostramos los resultados
    for epoch in range(1, epochs + 1):
        print('Epoch:', epoch)
        train_loss, train_accuracy = train(model, loss_criteria, optimizer, train_loader)
        test_loss, test_accuracy = test(model, loss_criteria, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        
    #mostramos la gráfica de la pérdida
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

    #evaluamos el modelo
    model.eval()
    print('Mostrando predicciones\n')
    
    truelabels = []
    predictions = []
    
    for data, target in test_loader:
        # Convertir tensores a matrices numpy
        data_np = data.numpy()
        target_np = target.numpy()
        
        # Pasar los datos a través del modelo
        output = model(data)
        _, predicted = torch.max(output, 1)
        
        # Convertir la salida y la etiqueta a matrices numpy
        truelabels.extend(target_np)
        predictions.extend(predicted.numpy()) 
    
    # Mostrar matriz de confusión
    cm = confusion_matrix(truelabels, predictions)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    clases = ['defensa', 'ataque', 'neutro']
    tick_marks = np.arange(len(clases))
    plt.xticks(tick_marks, clases, rotation=45)
    plt.yticks(tick_marks, clases)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.show()

   #guardamos el modelo
    modelo_ruta = os.path.join(os.path.dirname(__file__), 'modelo_red_neuronal_convolu_uefa.pth')
    torch.save(model.state_dict(), modelo_ruta)
    del model
    print('Modelo guardado en', modelo_ruta)
    
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



#Cargamos los datos
df_equipos = pd.read_csv('csvs/datos_fut_clasificados.csv', encoding='utf-8', delimiter=',')

#Eliminamos las variables categóricas
df_equipos = df_equipos.drop(['Club', 'Country'], axis=1)

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


#preparamos la base del modelo
model = torchvision.models.resnet18(pretrained=True)
#model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_ftrs = model.fc.in_features
for param in model.parameters():
    param.requires_grad = False

#Remplazamos la capa de predicción

num_classes = 3
model.fc = nn.Linear(num_ftrs, num_classes)

loss_criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

device = "cpu"
print('Entrenando en', device)
model = model.to(device)

def train(model, train_loader, optimizer, loss_criteria):
    # Establecer el modelo en modo de entrenamiento
    model.train()
    train_loss = 0.0
    
    # Procesar los datos en lotes
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
        # Restablecer el optimizador
        optimizer.zero_grad()
        
        #Ajustamos los datos para que se adapten a la entrada de la red
        inputs = inputs.view(inputs.size(0), 1, 1, -1) 
        inputs = inputs.repeat(1, 3, 1, 1)
        
        # Hacer pasar los datos a través de las capas del modelo
        outputs = model(inputs)
        
        # Ajustar dimensiones de las etiquetas
        labels = labels.view(-1)
        labels = labels - 1
        
        #calculamos la pérdida
        loss = loss_criteria(outputs, labels)
        # Mantener un total de pérdida
        train_loss += loss.item()
        
        # Retropropagar
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print('Conjunto de entrenamiento [{}/{} ({:.0f}%)] Pérdida: {:.6f}'.format(
                i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))
    avg_loss = train_loss / (i + 1)
    print('Conjunto de entrenamiento: Pérdida promedio: {:.6f}'.format(avg_loss))
    
    return avg_loss


def test(model, test_loader, loss_criteria):
    # Cambiar el modelo al modo de evaluación (para no retropropagar ni aplicar dropout)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        
        for data in test_loader:
          
            inputs, labels = data
            #Ajustamos los datos para que se adapten a la entrada de la red
            inputs = inputs.view(inputs.size(0), 1, 1, -1) 
            inputs = inputs.repeat(1, 3, 1, 1)
            
            # Hacer pasar los datos a través de las capas del modelo
            outputs = model(inputs)
            
            # Ajustar dimensiones de las etiquetas
            labels = labels.view(-1)
            labels = labels - 1
            
            # Obtener las clases predichas para este lote
            output = model(inputs)
            
            # Calcular la pérdida para este lote
            test_loss += loss_criteria(output, labels).item()
            
            # Calcular la precisión para este lote
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(labels == predicted).item()
            total += labels.size(0)

    # Calcular la pérdida promedio y la precisión total para esta época
    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / total
    
    print('Conjunto de validación: Pérdida promedio: {:.6f}, Precisión: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset), accuracy))
    
    return avg_loss, accuracy


if __name__ == '__main__':
    #inicializamos las listas para almacenar los resultados
    epoch_nums = []
    training_loss = []
    validation_loss = []

    #entrenamos el modelo
    epochs = 3
    for epoch in range(1, epochs + 1):
        print('Epoch:', epoch)
        train_loss = train(model, train_loader, optimizer, loss_criteria)
        test_loss, accuracy = test(model, test_loader, loss_criteria)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
    
    #PRECISIÓN Y VALIDACIÓN POBRE 
    
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
    

   #guardamos el modelo
    modelo_ruta = os.path.join(os.path.dirname(__file__), 'modelo_cnn_transf_uefa.pth')
    torch.save(model.state_dict(), modelo_ruta)
    del model
    print('Modelo guardado en', modelo_ruta)
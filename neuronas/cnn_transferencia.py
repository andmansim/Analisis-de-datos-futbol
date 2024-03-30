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

'''
para la cnn de transferencia adaptada  un csv tengo que crear la capa de predicción, donde una capa de predicción para el cnn de transferencia normal, es decir se procesan imagenes haría esto:
# Set the existing feature extraction layers to read-only
for param in model.parameters():
    param.requires_grad = False

# Replace the prediction layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

Cómo lo adapto yo para que funcione en los datos del csv?

'''
for param in model.parameters():
    param.requires_grad = False

#Remplazamos la capa de predicción
num_ftrs = model.fc.in_features
num_classes = 3
model.fc = nn.Linear(num_ftrs, num_classes)

'''
posteriromente hacemos la función train, donde le modelo de imagenes usa la siguiente.  
Me puedes ayudar a adaptarla para que funcione con los datos del csv?

def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)
        
        # Get the loss
        loss = loss_criteria(output, target)
        
        # Keep a running total
        train_loss += loss.item()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Print metrics for every 10 batches so we see some progress
        if batch_idx % 10 == 0:
            print('Training set [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
            
'''
def train(model, device, train_loader, optimizer, loss_criteria, epoch):
    # Establecer el modelo en modo de entrenamiento
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    
    # Procesar los datos en lotes
    for batch_idx, (data, target) in enumerate(train_loader):
        # Utilizar la CPU o GPU según corresponda
        data, target = data.to(device), target.to(device)
        
        # Restablecer el optimizador
        optimizer.zero_grad()
        
        # Hacer pasar los datos a través de las capas del modelo
        output = model(data)
        
        # Calcular la pérdida
        loss = loss_criteria(output, target)
        
        # Mantener un total de pérdida
        train_loss += loss.item()
        
        # Retropropagar
        loss.backward()
        optimizer.step()
        
        # Imprimir métricas para cada 10 lotes para ver el progreso
        if batch_idx % 10 == 0:
            print('Conjunto de entrenamiento [{}/{} ({:.0f}%)] Pérdida: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    # Calcular la pérdida promedio para la época
    avg_loss = train_loss / (batch_idx + 1)
    print('Conjunto de entrenamiento: Pérdida promedio: {:.6f}'.format(avg_loss))
    
    return avg_loss


'''
puedes hacer lo mismo con la función test?

def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss/batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss
'''




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
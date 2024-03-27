
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

import pandas as pd

#Cargamos los datos
df_equipos = pd.read_csv('csvs/datos_fut_clasificados.csv', encoding='utf-8', delimiter=',')

#Eliminamos las variables categóricas
df_equipos = df_equipos.drop(['Club', 'Country'], axis=1)

'''#Dividimos los datos en x, y
x = df_equipos.drop(['porganarpartido', 'porperderpartido', 'poremppartido'], axis=1)
y = df_equipos['categoria']

#dividimos los datos en train y test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
'''
#Dividimos los datos en x, y
x = df_equipos.drop(['porganarpartido', 'porperderpartido', 'poremppartido'], axis=1).values
y = df_equipos['categoria']

# Realizar cualquier preprocesamiento necesario y convertir los datos en tensores PyTorch
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)  # Suponiendo que 'categoria' es tu columna de etiquetas

#Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crear conjuntos de datos y cargadores de datos
train_dataset = td.TensorDataset(x_train, y_train)
test_dataset = td.TensorDataset(x_test, y_test)

batch_size = 32
train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('Datos cargados\n')

#creamos la red neuronal
class Net(nn.Module):
    #constructor
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
       
    
print('Red Neuronal creada\n')

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_criteria = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#entremaos el modelo
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    print('Epoch:', epoch)
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.unsqueeze(1))  # Agregar una dimensión de canal
        loss = loss_criteria(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)
    '''for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criteria(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_loss = train_loss / (batch_idx+1)
    print('Train set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss'''

#definimos la función de test
def test(model,test_loader,loss_criteria):
    #cambiamos el modelo a modo de evaluación
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            output = model(data.unsqueeze(1)) # Agregar una dimensión de canal
            
            test_loss += loss_criteria(output, target, reduction='sum').item() # sum up batch loss
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item() 
    avg_loss = test_loss / batch_count
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return avg_loss



if __name__ == '__main__':
    epoch_nums = []
    training_loss = []
    validation_loss = []

    #entrenamos el modelo
    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, loss_criteria, epoch)
        test_loss, test_acc = test(model, test_loader, loss_criteria)
        
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
        data, target = data.to(device), target.to(device)
        for label in target.cpu().data.numpy():
            truelabels.append(label)
        for prediction in model(data).cpu().data.numpy().argmax(1):
            predictions.append(prediction)

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
    model_file = 'formas_model.pth'
    torch.save(model.state_dict(), model_file)
    del model
    print('Modelo guardado en', model_file) 

    #Cargamos el modelo
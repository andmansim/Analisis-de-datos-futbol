
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

#Normalizamos los datos
df_equipos = (df_equipos - df_equipos.mean()) / df_equipos.std()

#Dividimos los datos en x, y
X = df_equipos.drop(['porganarpartido', 'porperderpartido', 'poremppartido'], axis=1).values
y = df_equipos['categoria']

# Realizar cualquier preprocesamiento necesario y convertir los datos en tensores PyTorch
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Agregar una dimensión de canal
y = torch.tensor(y, dtype=torch.long).unsqueeze(1)

#Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear conjuntos de datos y cargadores de datos
train_dataset = td.TensorDataset(X_train, y_train)
test_dataset = td.TensorDataset(X_test, y_test)

batch_size = 64 #indica el num de muestras de cada lote
train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('Datos cargados\n')

#creamos la red neuronal

'''class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.fc1 = nn.Linear(480, 128)
        self.fc2 = nn.Linear(128, 10)  # Suponiendo que tienes 10 clases de salida
        
    
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 1, 1)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 1, 1)
        
        x = x.view(-1, 480)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    '''
    
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
print('Red Neuronal creada\n')  

input_size = X.shape[1] #número de columnas de X
num_classes = len(pd.unique(y)) #número de clases diferentes (el número de variables que los queremos clasificar)

model = Net(input_size, num_classes)
loss_criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

#entremaos el modelo
def train(model, loss_criteria, optimizer, train_loader, epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    print('Epoch:', epoch)
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Ajustar dimensiones de las etiquetas
        labels = labels.view(-1)
        
        loss = loss_criteria(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    print('Train set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
        train_loss / len(train_loader.dataset), 100 * correct / total))
    return train_loss / len(train_loader.dataset), correct / total
    

def test(model, loss_criteria, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            
            # Ajustar dimensiones de las etiquetas
            labels = labels.view(-1)
            
            outputs = model(inputs)
            loss = loss_criteria(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
        test_loss / len(test_loader.dataset), accuracy))
    return test_loss / len(test_loader.dataset), accuracy

if __name__ == '__main__':
    epoch_nums = []
    training_loss = []
    validation_loss = []

    #entrenamos el modelo
    epochs = 5
    
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(model, loss_criteria, optimizer, train_loader, epoch)
        test_loss, test_accuracy = test(model, loss_criteria, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        
    '''  #mostramos la gráfica de la pérdida
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

    #Cargamos el modelo'''
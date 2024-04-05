
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#cogemos la carpeta img
data_path = os.path.join(os.path.dirname(__file__), 'img')

# Get the class names
classes = os.listdir(data_path)
classes.sort()
print(len(classes), 'classes:')
print(classes)

#Mostremos una imagen de cada clase
fig = plt.figure(figsize=(8, 12))
i = 0
for sub_dir in os.listdir(data_path):
    i+=1
    
    img_file = os.listdir(os.path.join(data_path,sub_dir))[0]
    img_path = os.path.join(data_path, sub_dir, img_file)
    img = mpimg.imread(img_path)
    a=fig.add_subplot(1, len(classes),i)
    a.axis('off')
    imgplot = plt.imshow(img)
    a.set_title(img_file)
plt.show()

# Función para cargar el conjunto de datos
def load_dataset(data_path):
    # Cargar todas las imágenes, transformándolas
    # a tensores y normalizándolas con una media y desviación estándar específicas
    #transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transformation = transforms.Compose([
        transforms.Resize((225, 225)),  # Redimensiona todas las imágenes a 225x225
        transforms.ToTensor(),  # Convierte las imágenes a tensores
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normaliza las imágenes
    ])
    # Cargar todas las imágenes, transformamos
    full_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=transformation)
    
    #Dividimos en conjuntos de entrenamiento (70%) y prueba (30%)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Preparamos los dataloaders para entrenamiento y prueba
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )
    
    # Preparamos los dataloaders para prueba
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )
        
    return train_loader, test_loader

#Separamos los datos en train y test
train_loader, test_loader = load_dataset(data_path)
print('Datos listos para usar\n')

#Creamos la red neuronal 
class Net(nn.Module):
    # Constructor
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        #creamos las capas de la red
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        
        self.drop = nn.Dropout2d(p=0.2)

        self.fc = nn.Linear(in_features=75264, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.drop(self.conv3(x)))
        x = F.dropout(x, training=self.training)
        #Aplanamos los datos        
        x = x.view(x.size(0), -1)
      
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    
print('Clase cnn lista\n')

def train(model, device, train_loader, optimizer, epoch):
    #Entrenamos el modelo
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Procesamos las imágenes en lotes
    for batch_idx, (data, target) in enumerate(train_loader):
        # Enviamos los datos al dispositivo
        data, target = data.to(device), target.to(device)
        # Reseteamos el optimizador
        optimizer.zero_grad()
        # Pasamos los datos a través del modelo
        output = model(data)
        # Obtenemos la pérdida
        loss = loss_criteria(output, target)
        # Mantenemos un total en ejecución
        train_loss += loss.item()
        # Realizamos la retropropagación
        loss.backward()
        optimizer.step()
        
        # Mostramos las métricas para cada 10 lotes para ver el progreso
        if batch_idx % 10 == 0:
            print('Training set [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # Devolvemos la pérdida promedio para la época
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
            
            
def test(model, device, test_loader):
    # Evaluamos el modelo
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            #Con el modelo, predecimos las clases
            output = model(data)
            
            #Calculamos la perdida de cada lote          
            test_loss += loss_criteria(output, target).item()
            #Obtenemos el número de predicciones correctas
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    #Calculamos la perdida promedio y la precisión total para la época
    avg_loss = test_loss/batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return avg_loss
    
    

device = "cpu"
if (torch.cuda.is_available()):
    # Si hay GPU disponible, usamos cuda
    device = "cuda"
print('Training on', device)
# Creamos una instancia de la clase del modelo y la asignamos al dispositivo
model = Net(num_classes=len(classes)).to(device)

#Usamos el optimizador Adam para ajustar los pesos
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Especificamos el criterio de pérdida
loss_criteria = nn.CrossEntropyLoss()

#Listas para almacenar las métricas
epoch_nums = []
training_loss = []
validation_loss = []

#Entrenamos durante 5 épocas (en un escenario real, probablemente usarías muchas más)
epochs = 5
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

'''
Vamos a entrenar una CNN(red neuronal convolucional) capaz de clasificar
imagenes de formas geométricas simples. 
'''

#Importamos las librerías
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.autograd as Variable
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#ns yo copio y si se cambia pues se cambia después
#ponemos el directorio de trabajo(donde están las img)ç
data_path = 'imagenes'
#creamos un objeto que nos permita cargar las imágenes
#Buscamos y ordenamos todas las posibles clases de las formas de las img
clases = os.listdir(data_path)
clases.sort()
print(len(clases), 'clases:', clases)

#mostramos cada imagen de la carpeta
fig = plt.figure(figsize=(10, 10))
i = 0
for sub_dir in os.listdir(data_path):
    i+=1
    img_file = os.listdir(os.path.join(data_path, sub_dir))[0]  
    img_path = os.path.join(data_path, sub_dir, img_file) 
    img = mpimg.imread(img_path)
    a = fig.add_subplot(1, len(clases), i)
    a.axis('off')
    imgplot = plt.imshow(img)
    a.set_title(img_file)
plt.show()

#cargamos y transformamos las img en tensores, son los datos de entrada 
#los vamos a normalizar y usarlos en la escala 0.5

def load_dataset(data_path):
    #funcion que carga las imagenes y las transforma en tensores
    transformacion = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) ])
    
    #cargamos las imagenes y las transformamos
    full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transformacion)
    
    #70% test y 30% train
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])          
    
    #definimos los dataloaders para el train y test
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, num_workers=0, shuffle=True) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, num_workers=0, shuffle=True)   
    return train_loader, test_loader

train_loader, test_loader = load_dataset(data_path)
print('Datos cargados y transformados en tensores\n')

#creamos la red neuronal
class Net(nn.Module):
    #constructor
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        #definimos las capas de la red
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        
        self.drop = nn.Dropout2d(p=0.2)
        
        self.fc = nn.Linear(in_features=32*32*24, out_features=num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(x))

        x = F.dropout(x, training=self.training)
        
        x = x.view(-1, 32*32*24)

        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
print('Red Neuronal creada\n')

#entremaos el modelo
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    print('Epoch:', epoch)
    
    for batch_idx, (data, target) in enumerate(train_loader):
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
    return avg_loss

#definimos la función de test
def test(model, device, test_loader):
    #cambiamos el modelo a modo de evaluación
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            test_loss += loss_criteria(output, target, reduction='sum').item() # sum up batch loss
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item() 
    avg_loss = test_loss / batch_count
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return avg_loss

#definimos el dispositivo
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print('Usando:', device)

#creamos el modelo
model = Net(num_classes=len(clases)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_criteria = nn.CrossEntropyLoss()

epoch_nums = []
training_loss = []
validation_loss = []

#entrenamos el modelo
epochs = 5
for epoch in range(1, epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    test_loss = test(model, device, test_loader)
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
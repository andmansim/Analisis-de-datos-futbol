
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
    transformacion = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) ])
    
    #cargamos las imagenes y las transformamos
    full_dataset = torchvision.datasets.ImageFolder( 
        root=data_path, transform=transformacion)
    
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
    
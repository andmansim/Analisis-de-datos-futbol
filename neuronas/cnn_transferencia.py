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
print(model)


# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as td
import torchvision.transforms as transforms

# Cargar los datos desde el CSV
df_equipos = pd.read_csv('csvs/datos_fut_clasificados.csv', encoding='utf-8', delimiter=',')

# Eliminar las variables categóricas
df_equipos = df_equipos.drop(['Club', 'Country'], axis=1)

# Obtener las coordenadas X e Y de tus datos
x_coords = df_equipos['coord_x'].values
y_coords = df_equipos['coord_y'].values

# Crear un histograma bidimensional (mapa de calor) de los eventos en el campo de fútbol
heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=20)

# Plotear el mapa de calor
plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest')
plt.colorbar(label='Número de eventos')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Mapa de calor de eventos en el campo de fútbol')
plt.show()

# Definir una CNN preentrenada (por ejemplo, ResNet)
model = models.resnet18(pretrained=True)

# Congelar los pesos de la red preentrenada
for param in model.parameters():
    param.requires_grad = False

# Añadir una capa lineal para procesar el mapa de calor
class HeatmapLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HeatmapLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Definir las dimensiones de entrada y salida de la capa de mapa de calor
input_size_heatmap = 20 * 20  # Suponiendo que tienes un mapa de calor de 20x20
output_size_heatmap = 64  # Suponiendo un tamaño arbitrario para la salida

# Crear el modelo combinando la CNN preentrenada y la capa de mapa de calor
model_heatmap = nn.Sequential(model, HeatmapLayer(input_size_heatmap, output_size_heatmap))

# Dividir el mapa de calor en conjuntos de entrenamiento y prueba
X_train_heatmap, X_test_heatmap = train_test_split(heatmap, test_size=0.2, random_state=42)

# Dividir las características originales (si las tienes) en conjuntos de entrenamiento y prueba
X_train_original, X_test_original = train_test_split(df_equipos.drop(['coord_x', 'coord_y'], axis=1), test_size=0.2, random_state=42)

# Concatenar el mapa de calor con las características originales (si las tienes)
X_train_combined = np.concatenate((X_train_original, X_train_heatmap.reshape(-1, input_size_heatmap)), axis=1)
X_test_combined = np.concatenate((X_test_original, X_test_heatmap.reshape(-1, input_size_heatmap)), axis=1)

# Supongamos que 'y_train' y 'y_test' son tus etiquetas objetivo
y_train = df_equipos['categoria'].values
y_test = df_equipos['categoria'].values

# Convertir los datos combinados a tensores de PyTorch
X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Crear conjuntos de datos y cargadores de datos para la CNN
train_dataset = td.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = td.TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Ahora puedes entrenar tu modelo combinado usando train_loader y test_loader


# Entrenar tu modelo con los datos combinados de la imagen y el mapa de calor
# (recuerda concatenar los datos de imagen y mapa de calor como entrada)
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
    
    
# Now use the train and test functions to train and test the model    

device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"
print('Training on', device)

# Create an instance of the model class and allocate it to the device
model = model.to(device)

# Use an "Adam" optimizer to adjust weights
# (see https://pytorch.org/docs/stable/optim.html#algorithms for details of supported algorithms)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Specify the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 3 epochs (in a real scenario, you'd likely use many more)
epochs = 3
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

'''
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

y_train_adjusted = y_train - 1
y_test_adjusted = y_test - 1

train_dataset = td.TensorDataset(X_train, y_train_adjusted)
test_dataset = td.TensorDataset(X_test, y_test_adjusted)

#creamos los cargadores de datos
batch_size = 64 #indica el num de muestras de cada lote
train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('Datos cargados\n')

#creamos la capa de predicción
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False
    
#remplazamos la capa de predicción
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

#mostramos el modelo
print(model)

#entremaos el modelo
def train(model, loss_criteria, optimizer, train_loader):
    #activamos el modo de entrenamiento
    model.train()
    
    #inicializamos las variables
    train_loss = 0.0
 
    for i, data in enumerate(train_loader):
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
        
        if i % 10 == 0:
            print('training set [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i * len(inputs), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))
        
    #mostramos el promedio de la pérdida y la precisión
    average_loss = train_loss / len(train_loader.dataset)
    print('Train set: Average loss: {:.4f}'.format(average_loss))
    return average_loss
    

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
    average_loss = test_loss / len(test_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
        average_loss, 100 * correct / total))
    
    return average_loss



#Main
#definimos el tamaño de la entrada y el número de clases
input_size = X.shape[1] #número de columnas de X
num_classes = 3 #número de clases diferentes (el número de variables que los queremos clasificar)


#definimos la función de pérdida y el optimizador
loss_criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


if __name__ == '__main__':
    #inicializamos las listas para almacenar los resultados
    epoch_nums = []
    training_loss = []
    validation_loss = []

    #entrenamos el modelo
    epochs = 3
    
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
    '''
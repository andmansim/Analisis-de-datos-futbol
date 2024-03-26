
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

#

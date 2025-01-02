import pandas as pd
import numpy as np
import h5py
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from torch import nn, optim
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Definir variables globales
ancho = 32
alto = 32
canales = 3
pixeles = ancho*alto*canales
cantidadClases = 10
clases = ['Avión', 'Carro', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']
batchSize = 100
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
current_dir = os.getcwd()

#Transformaciones entrenamiento
train_transform = transforms.Compose(
  [transforms.Resize((ancho, alto)), # Resize the image in a 32X32 shape
  transforms.RandomRotation(20), # Randomly rotate some images by 20 degrees
  transforms.RandomHorizontalFlip(0.1), # Randomly horizontal flip the images
  transforms.ColorJitter(
    brightness = 0.1, # Randomly adjust color jitter of the images
    contrast = 0.1, 
    saturation = 0.1), 
  transforms.RandomAdjustSharpness(
    sharpness_factor = 2,
    p = 0.1), # Randomly adjust sharpness
  transforms.ToTensor(),   # Converting image to tensor
  transforms.Normalize(mean, std), # Normalizing with standard mean and standard deviation
  transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)]
)

#Transformaciones prueba
test_transform = transforms.Compose(
  [transforms.Resize((ancho,alto)),
  transforms.ToTensor(),
  transforms.Normalize(mean, std)]
  )

#Datos de Entrenamiento
dataset_path = os.path.join(current_dir, 'dataset','train')
trainset = CIFAR10(root=dataset_path, train=True, download=False, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True)

#Datos de Prueba
testset_path = os.path.join(current_dir, 'dataset','test')
testset = CIFAR10(root=testset_path,train=False, download=False, transform=test_transform)
testloader = DataLoader(testset, batch_size=batchSize, shuffle=True)

#Función auxiliar para graficar probabilidad
def view_classify(img, ps):
  ps = ps.data.numpy().squeeze()
  fig, (ax1, ax2) = plt.subplots(figsize=(16,13), ncols=2)
  ax1.imshow(img.numpy().transpose((1, 2, 0)))
  ax1.axis('off')
  ax2.barh(np.arange(cantidadClases), ps)
  ax2.set_aspect(0.1)
  ax2.set_yticks(np.arange(cantidadClases))
  ax2.set_yticklabels(clases, size='medium');
  ax2.set_title('Probabilidad de las Clases')
  ax2.set_xlim(0, 1.1)

  plt.tight_layout()
  
#Definir estructura y parámetros del modelo
print("Fin")



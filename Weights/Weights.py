##### codigo pasado a python y modificado por chat##########

import os
import json
import numpy as np
import torch
from torchvision import transforms as T, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Configuración de transformaciones
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar datasets
train_data_path = os.path.join('dataset', 'train')
test_data_path = os.path.join('dataset', 'test')

train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
val_dataset = ImageFolder(root=train_data_path, transform=val_transform)
test_dataset = ImageFolder(root=test_data_path, transform=val_transform)

# Dividir dataset de entrenamiento en entrenamiento y validación
val_size = 0.2
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(val_size * num_train))
random_seed = 42
np.random.seed(random_seed)
np.random.shuffle(indices)
train_idx, val_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# Crear DataLoaders
BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

# Configurar modelo
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

n_inputs = model.fc.in_features
n_outputs = len(train_dataset.classes)  # Ajustar según el número de clases en el dataset

# Redefinir capa final
sequential_layers = nn.Sequential(
    nn.Linear(n_inputs, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, n_outputs),
    nn.LogSoftmax(dim=1)
)
model.fc = sequential_layers
model = model.to(device)

# Configurar criterio y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Entrenamiento del modelo
loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
EPOCHS = 15

for epoch in range(1, EPOCHS + 1):
    best_acc = 0.0
    print(f"\nEpoch {epoch}/{EPOCHS}\n{'='*25}")
    for phase in ['train', 'val']:
        running_loss = 0.0
        running_corrects = 0
        if phase == 'train':
            model.train()
        else:
            model.eval()
        for inputs, labels in loaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        if phase == 'train':
            scheduler.step()
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = deepcopy(model.state_dict())
        print(f"Loss ({phase}): {epoch_loss:.4f}, Acc ({phase}): {epoch_acc:.4f}")

# Guardar el mejor modelo
torch.save(best_model_weights, '../model/foodnet_resnet18.pth')


#################### No Usar Pesos Preentrenados: ########################

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from copy import deepcopy

# Ruta de los datos
data_path = 'dataset'

# Rutas para entrenamiento y prueba
train_data_path = os.path.join(data_path, 'train')
test_data_path = os.path.join(data_path, 'test')

# Definir transformaciones
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Crear datasets
train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
val_dataset = ImageFolder(root=train_data_path, transform=val_transform)
test_dataset = ImageFolder(root=test_data_path, transform=val_transform)

# Dividir en conjuntos de entrenamiento y validación
val_size = 0.2
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(val_size * num_train))

random_seed = 42
np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, val_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# Crear DataLoaders
BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

# Definir el modelo sin pesos preentrenados
import torchvision.models as models

model = models.resnet18(pretrained=False)  # No usar pesos preentrenados

# Modificar la última capa para adaptarse a tu número de clases
n_inputs = model.fc.in_features
n_outputs = len(train_dataset.classes)  # Número de clases

sequential_layers = nn.Sequential(
    nn.Linear(n_inputs, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, n_outputs),
    nn.LogSoftmax(dim=1)
)

model.fc = sequential_layers
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Definir el criterio y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Entrenamiento
loaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}

EPOCHS = 15

for epoch in range(1, EPOCHS + 1):
    best_acc = 0.0
    print(f"\nEpoch {epoch}/{EPOCHS}\n{'='*25}")
    for phase in ['train', 'val']:
        running_loss = 0.0
        running_corrects = 0
        if phase == 'train':
            model.train()
        else:
            model.eval()
        for inputs, labels in loaders[phase]:
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        if phase == 'train':
            scheduler.step()
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = deepcopy(model.state_dict())
        print(f"Loss ({phase}): {epoch_loss}, Acc ({phase}): {epoch_acc}")

# Guardar el modelo con las mejores ponderaciones
torch.save(best_model_weights, '../model/foodnet_resnet18.pth')

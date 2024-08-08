# -*- coding: utf-8 -*-
"""
@author: SSALAZAR
"""

import segmentation_models_pytorch as smp

import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold
import albumentations as album

from tqdm import tqdm


# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

class RoadsDataset(torch.utils.data.Dataset):

    """Roads Dataset. Read images, apply augmentation and preprocessing transformations.
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline (flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing (normalization, shape manipulation, etc."""

    def __init__(
            self, 
            images_dir, 
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,):
      
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)#ORIGINAL
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image#, mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor))
        
    return album.Compose(_transform)

def predict(CLASS_PREDICT = ['Vegetacion', 'CuerposAgua', 'Construcciones', 'Vias', 'Otros'],SIZE=100, CLASS_RGB=[[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0], [128, 128, 128]],  # RGB para las 5 clases] 
            CROP_PATH='./crop_images/',ENCODER = 'vgg16', ENCODER_WEIGHTS = 'imagenet', ACTIVATION = 'sigmoid',WEIGHT='./weight/Road_weight_k0_e25_vgg16.pth', PRED_PATH = './prediction/predictions/', CLASS_INDEX=None, clean_crop=False):

    #ajustar dimensiones de la imagen
    height_crop=width_crop=(SIZE//32)*32
    height_pad=width_pad=((SIZE//32)*32)+32
    
    # Aumentacion para la validacion
    def get_validation_augmentation():   
        test_transform = [
            album.PadIfNeeded(min_height=height_pad, min_width=width_pad, always_apply=True, border_mode=0),
        ]
        return album.Compose(test_transform)

    # Definición de clases y valores RGB
    class_names = CLASS_PREDICT
    class_rgb_values = CLASS_RGB

    # Seleccionar clases
    select_classes = CLASS_PREDICT
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    # Directorio de datos
    DATA_DIR = CROP_PATH
    x_test_dir = os.path.join(DATA_DIR, "images")

    # Crear el modelo de segmentación con un encoder preentrenado
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(select_classes),  # Asegurarse de que esté configurado para 5 clases
        activation=ACTIVATION,
    )

    # Obtener la función de preprocesamiento correspondiente
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Crear el dataloader de prueba
    test_dataset = RoadsDataset(
        x_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    # Configurar el dispositivo: `CUDA` o `CPU`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar el mejor modelo guardado
    if os.path.exists(WEIGHT):
        best_model = torch.load(WEIGHT, map_location=DEVICE)
        print('Modelo UNet cargado exitosamente.')

    # Recortar la imagen centrada para devolverla a las dimensiones originales
    def crop_image(image, target_image_dims=[100, 100, 3]):
        target_size = target_image_dims[0]
        image_size = len(image)
        padding = (image_size - target_size) // 2

        if padding < 0:
            return image

        return image[
            padding:image_size - padding,
            padding:image_size - padding, :, 
        ]

    # Crear carpeta para las predicciones si no existe
    sample_preds_folder = PRED_PATH
    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)

    # Realizar predicciones para cada imagen en el dataset de prueba
    for idx, name in tqdm(zip(range(len(test_dataset)), [i.split('/')[-1] for i in test_dataset.image_paths])):
        image = test_dataset[idx]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

        # Predecir la máscara de la imagen de prueba
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()

        # Convertir la predicción de `CHW` a `HWC`
        pred_mask = np.transpose(pred_mask, (1, 2, 0))

        # Convertir la predicción a la clase correspondiente
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))

        # Guardar la imagen de predicción
        cv2.imwrite(os.path.join(sample_preds_folder, name), np.hstack([pred_mask]))

        # Borrar cortes después de predecirlos si está configurado
        if clean_crop:
            os.remove(os.path.join(x_test_dir, name))
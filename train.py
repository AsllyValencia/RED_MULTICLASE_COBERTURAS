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
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold
import albumentations as album


# Definición de la función de pérdida combinada
class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super(DiceCrossEntropyLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, outputs, targets):
        dice_loss = smp.utils.losses.DiceLoss()(outputs, targets)
        ce_loss = F.cross_entropy(outputs, targets)
        return self.weight_dice * dice_loss + self.weight_ce * ce_loss


# Valores RGB para cada clase
label_values = [
    [0, 255, 0],    # Vegetacion
    [0, 0, 255],    # CuerposAgua
    [255, 0, 0],    # Construccion
    [255, 255, 0],  # Vias
    [128, 128, 128] # Otros
]

def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values: List of RGB values for each class
        
    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of num_classes
    """
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
        label_values: List of RGB values for each class

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)  # Convert label values to a numpy array
    x = colour_codes[image.astype(int)]   # Map each class index to its corresponding RGB color

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
            masks_dir, 
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,):
      
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        #self.class_rgb_values = [self.CLASSES.index(cls) for cls in class_names]
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        
        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
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
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)

#random_idx = random.randint(0, len(augmented_dataset)-1)

# Función principal de entrenamiento
def train(TRAIN_VALID="./data/", SIZE=100, CLASS_PREDICT = ['background', 'road'], CLASS_RGB=[[0, 0, 0], [255, 255, 255]], ENCODER = 'resnet50', ENCODER_WEIGHTS = 'imagenet', ACTIVATION = 'sigmoid', EPOCHS = 5, LEARNING_R=0.0001,BATCH_TRAIN=16,BATCH_VALID=1, ds='dataset'):
    
    height_crop=width_crop=(SIZE//32)*32
    height_pad=width_pad=((SIZE//32)*32)+32
    
    def get_training_augmentation():
        train_transform = [
            album.RandomCrop(height=height_crop, width=width_crop, always_apply=True),
            album.OneOf(
                [
                    album.HorizontalFlip(p=1),
                    album.VerticalFlip(p=1),
                    album.RandomRotate90(p=1),
                ],
                p=0.75,
            ),
        ]
        return album.Compose(train_transform)


    def get_validation_augmentation():   
        # Add sufficient padding to ensure image is divisible by 32
        test_transform = [
            album.PadIfNeeded(min_height=height_pad, min_width=width_pad, always_apply=True, border_mode=0),
        ]
        return album.Compose(test_transform)
    
    #DATA SOURCE
    DATA_DIR = TRAIN_VALID

    x_train_dir = os.path.join(DATA_DIR, "train/images")
    y_train_dir = os.path.join(DATA_DIR, "train/masks")

    x_valid_dir = os.path.join(DATA_DIR, "valid/images")
    y_valid_dir = os.path.join(DATA_DIR, "valid/masks")

    class_names = CLASS_PREDICT
    class_rgb_values =CLASS_RGB

    # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = CLASS_PREDICT
    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    augmented_dataset=RoadsDataset(
    x_train_dir, y_train_dir, 
    augmentation=get_training_augmentation(),
    class_rgb_values=select_class_rgb_values,)

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, 
        classes=5, # 5 canales de salida para cada clase
        activation=ACTIVATION,)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Get train and val dataset instances
    train_dataset = RoadsDataset(
        x_train_dir, y_train_dir, augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,)

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_TRAIN, shuffle=True, 
                            num_workers=0) #En windows puede presentarse errores con valores diferentes a 0

    # Get train and val dataset instances
    valid_dataset = RoadsDataset(
        x_valid_dir, y_valid_dir, augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,)

    # Get train and val data loaders
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_VALID, shuffle=False, 
                            num_workers=0) #En windows puede presentarse errores con valores diferentes a 0

    # Set device: `CUDA` or `CPU`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Definir la función de pérdida combinada
    loss = DiceCrossEntropyLoss(weight_dice=0.5, weight_ce=0.5)
    
    # define METRICS
    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    
    # define OPTIMIZER
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=LEARNING_R),])


    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5,)

    # load best saved model checkpoint from previous commit (if present)
    if os.path.exists('./weight/model_weight.pth'):
        model = torch.load('./weight/model_weight.pth', map_location=DEVICE)

    train_epoch = smp.utils.train.TrainEpoch(
        model,     loss=loss,     metrics=metrics, 
        optimizer=optimizer,    device=DEVICE,
        verbose=True,)

    valid_epoch = smp.utils.train.ValidEpoch(
        model,     loss=loss,     metrics=metrics, 
        device=DEVICE,    verbose=True,)

    # TRAINING
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []
    for i in range(0, EPOCHS):
        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, './weight_{}_{}_{}_{}_ep{}_batch_{}_{}_lr_{}.pth'.format(ds,ENCODER, ENCODER_WEIGHTS, ACTIVATION, EPOCHS, BATCH_TRAIN,BATCH_VALID,str(LEARNING_R).split(".")[1]))
            print('Model saved!')
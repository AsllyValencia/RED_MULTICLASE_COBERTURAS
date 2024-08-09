
import pandas as pd
import os
from osgeo import gdal
#import gdal
from PIL import Image
from tqdm import tqdm_notebook as tqdmn
import cv2


def crop(ORI_PATH="./data/big_img/",SIZE=100,CROP_PATH='crop_images/images', EXT='tif'):

    original_path = ORI_PATH
    path_out = os.path.join(CROP_PATH)

    if not os.path.exists(CROP_PATH): # here '100' corresponds to size by default
        os.makedirs(CROP_PATH)

    list_imgs_original = [img_ori for img_ori in os.listdir(original_path)]

    for l in tqdmn(list_imgs_original):
        #img = Image.open(os.path.join(original_path, l))
        #h_img = img.size[0]
        #w_img = img.size[1]
        
        img = cv2.imread(os.path.join(original_path, l))
        h_img = img.shape[0]
        w_img = img.shape[1]

        img_width, img_height, dimension = img.shape
        
        height_sizes = [SIZE]
        width_sizes =  [SIZE]
        
        for height in height_sizes:
            width=height
            k = 0
            for i in range(0, img_height, height):
                for j in range(0, img_width, width):
                    try:
                        
                        imagen_test_name = os.path.join(path_out, l.replace("."+EXT, '') + '_{}_{}_{}_{}_{}_1_gdal.{}'\
                                        .format(i, j, k, height, width,EXT))
                        
                        #print(imagen_test_name)
                        if not os.path.exists(imagen_test_name):
                                            
                            # USING GDAL TO PROCESS UINT16 values on 
                            gdal.Translate( imagen_test_name, os.path.join(original_path, l.replace(EXT, EXT)),  
                                            options='-srcwin {} {} {} {}'.format(j, i, width, height))
                        #break
                                
                    except:
                        pass
                    k += 1


############# codigo modificado apra mas bandas#############

import os
from osgeo import gdal
from tqdm import tqdm
import cv2

def crop(ORI_PATH="./data/big_img/", SIZE=100, CROP_PATH='crop_images/images', EXT='tif'):
    """
    Cortar imágenes grandes en piezas más pequeñas usando OpenCV y GDAL.
    
    # Argumentos
        ORI_PATH: Ruta a las imágenes originales.
        SIZE: Tamaño de las piezas cortadas.
        CROP_PATH: Ruta para guardar las imágenes cortadas.
        EXT: Extensión de las imágenes.
    """

    original_path = ORI_PATH
    path_out = os.path.join(CROP_PATH)

    if not os.path.exists(CROP_PATH):
        os.makedirs(CROP_PATH)

    list_imgs_original = [img_ori for img_ori in os.listdir(original_path) if img_ori.endswith(EXT)]

    for l in tqdm(list_imgs_original):
        img = cv2.imread(os.path.join(original_path, l))
        h_img = img.shape[0]
        w_img = img.shape[1]
        img_width, img_height, dimension = img.shape

        height_sizes = [SIZE]
        width_sizes = [SIZE]

        for height in height_sizes:
            width = height
            k = 0
            for i in range(0, img_height, height):
                for j in range(0, img_width, width):
                    try:
                        imagen_test_name = os.path.join(path_out, l.replace(".{}".format(EXT), '') + '_{}_{}_{}_{}_{}_1_gdal.{}'.format(i, j, k, height, width, EXT))
                        
                        if not os.path.exists(imagen_test_name):
                            # Usar GDAL para procesar valores UINT16 y cortar la imagen
                            gdal.Translate(imagen_test_name, os.path.join(original_path, l),
                                           options='-srcwin {} {} {} {}'.format(j, i, width, height))
                    except Exception as e:
                        print(f"Error processing {l}: {e}")
                    k += 1


###################### codigo modificado para opsiones de multiples bandas############ 

from osgeo import gdal
import numpy as np
import os, glob
from tqdm import tqdm

def crop(ORI_PATH="./data/big_img/", SIZE=100, CROP_PATH='crop_images/images', EXT='tif'):
    """
    Cortar imágenes grandes en piezas más pequeñas.
    
    # Argumentos
        ORI_PATH: Ruta a las imágenes originales.
        SIZE: Tamaño de las piezas cortadas.
        CROP_PATH: Ruta para guardar las imágenes cortadas.
        EXT: Extensión de las imágenes.
    """

    original_path = ORI_PATH
    fol_crop = os.path.join(CROP_PATH)

    if not os.path.exists(CROP_PATH):
        os.makedirs(CROP_PATH)

    list_imgs_original = glob.glob(original_path + '/*.{}'.format(EXT))

    for im in tqdm(list_imgs_original):
        ds = gdal.Open(im)

        num_bands = ds.RasterCount
        data = []
        for i in range(num_bands):
            band = ds.GetRasterBand(i + 1)
            band_data = band.ReadAsArray()
            data.append(band_data)
        data = np.array(data)
        img_width, img_height, dimension = data.shape
        height_sizes = [SIZE]
        width_sizes = [SIZE]

        for height in height_sizes:
            width = height
            k = 0
            for i in range(0, img_height, height):
                for j in range(0, img_width, width):
                    try:
                        imagen_test_name = os.path.join(fol_crop, im.split('/')[-1].replace(".{}".format(EXT), '') + '_{}_{}_{}_{}_{}_1.{}'.format(i, j, k, height, width, EXT))
                        print(imagen_test_name)
                        if not os.path.exists(imagen_test_name):
                            gdal.Translate(imagen_test_name, im, options='-srcwin {} {} {} {}'.format(j, i, width, height))
                    except Exception as e:
                        print(f"Error processing {im}: {e}")
                    k += 1

import rasterio
from rasterio.transform import from_origin
def georeference(big_tif,ori_tif,geo_tif):
  # Abrir la imagen de referencia para obtener su transformación y sistema de coordenadas
  with rasterio.open(ori_tif) as src_referencia:
      transform_referencia = src_referencia.transform
      crs_referencia = src_referencia.crs

  # Abrir la imagen a georreferenciar para obtener sus dimensiones
  with rasterio.open(big_tif) as src_a_georreferenciar:
      width, height = src_a_georreferenciar.width, src_a_georreferenciar.height

      # Crear una nueva imagen georreferenciada
      with rasterio.open(geo_tif, 'w',
                        driver='GTiff', width=width, height=height,
                        count=src_a_georreferenciar.count,
                        dtype=src_a_georreferenciar.dtypes[0],
                        crs=crs_referencia, transform=transform_referencia) as dst_georreferenciada:
          for band in range(1, src_a_georreferenciar.count + 1):
              data = src_a_georreferenciar.read(band)
              dst_georreferenciada.write(data, band)

  print("Proceso de georreferenciación completado.")


##########CODIGO MODIFICADOOO #######

import rasterio
from rasterio.transform import from_origin

def georeference(big_tif, ori_tif, geo_tif):
    """
    Georreferencia una imagen raster utilizando la información de una imagen de referencia.

    # Argumentos
        big_tif: Ruta del archivo TIFF a georreferenciar.
        ori_tif: Ruta del archivo TIFF de referencia (con información de georreferenciación).
        geo_tif: Ruta del archivo TIFF de salida georreferenciado.
    """
    try:
        # Abrir la imagen de referencia para obtener su transformación y sistema de coordenadas
        with rasterio.open(ori_tif) as src_referencia:
            transform_referencia = src_referencia.transform
            crs_referencia = src_referencia.crs

        # Abrir la imagen a georreferenciar para obtener sus dimensiones
        with rasterio.open(big_tif) as src_a_georreferenciar:
            width, height = src_a_georreferenciar.width, src_a_georreferenciar.height

            # Crear una nueva imagen georreferenciada
            with rasterio.open(geo_tif, 'w',
                              driver='GTiff', width=width, height=height,
                              count=src_a_georreferenciar.count,
                              dtype=src_a_georreferenciar.dtypes[0],
                              crs=crs_referencia, transform=transform_referencia) as dst_georreferenciada:
                for band in range(1, src_a_georreferenciar.count + 1):
                    data = src_a_georreferenciar.read(band)
                    dst_georreferenciada.write(data, band)

        print("Proceso de georreferenciación completado.")

    except Exception as e:
        print(f"Error durante la georreferenciación: {e}")

# Ejemplo de uso
# georeference('./big_image.tif', './reference_image.tif', './georeferenced_image.tif')

import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

def raster_to_shapefile(input_raster, output_shapefile):
    # Read the raster
    with rasterio.open(input_raster) as src:
        # Read the raster data
        band = src.read(1)

        # Convert the raster to vector shapes
        vector_shapes = list(shapes(band, mask=None, transform=src.transform))

        # Create a geopandas GeoDataFrame from the shapes
        records = []
        for geom, value in vector_shapes:
            if value == 0:  # Modify this condition based on your road value
                continue
            records.append({'geometry': shape(geom)})

        gdf = gpd.GeoDataFrame(records)
        
        # Set the active geometry column
        gdf.set_geometry('geometry', inplace=True)

        # Assign the CRS to the GeoDataFrame
        gdf.crs = src.crs

        # Save the GeoDataFrame as a shapefile
        gdf.to_file(output_shapefile)

##################################MODIFICADO APRA 5 CLASES#####################################

        import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

def raster_to_shapefile(input_raster, output_shapefile):
    """
    Convierte un archivo raster en un shapefile.
    
    # Argumentos
        input_raster: Ruta del archivo raster de entrada.
        output_shapefile: Ruta del archivo shapefile de salida.
    """
    try:
        # Leer el raster
        with rasterio.open(input_raster) as src:
            # Leer los datos del raster
            band = src.read(1)

            # Convertir el raster en formas vectoriales
            vector_shapes = list(shapes(band, mask=None, transform=src.transform))

            # Crear un GeoDataFrame de geopandas a partir de las formas
            records = []
            for geom, value in vector_shapes:
                if value != 0:  # Modificar esta condición si es necesario para tus clases
                    records.append({'geometry': shape(geom), 'value': value})

            # Crear el GeoDataFrame
            gdf = gpd.GeoDataFrame(records, geometry='geometry')

            # Asignar el CRS al GeoDataFrame
            gdf.crs = src.crs

            # Guardar el GeoDataFrame como un shapefile
            gdf.to_file(output_shapefile)

        print(f"Shapefile guardado en: {output_shapefile}")

    except Exception as e:
        print(f"Error durante la conversión: {e}")

# Ejemplo de uso
# raster_to_shapefile('./input_raster.tif', './output_shapefile.shp')

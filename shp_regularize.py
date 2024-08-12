
##################### Post-procesamiento y Guardado de Shapefiles ##################

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os

def shp_regularize_multi_class(shp_in, output_dir, simp_fac, topol):
    # Leer el shapefile de entrada
    gdf = gpd.read_file(shp_in)
    
    # Verificar que la columna 'class' exista
    if 'class' not in gdf.columns:
        raise ValueError("El shapefile debe contener una columna 'class' para identificar las coberturas.")
    
    # Obtener las clases únicas
    classes = gdf['class'].unique()
    
    # Asegurarse de que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)
    
    for class_id in classes:
        # Filtrar el GeoDataFrame por la clase actual
        gdf_class = gdf[gdf['class'] == class_id]
        
        # Lista para almacenar geometrías regularizadas
        regularized_geometries = []

        for index, row in gdf_class.iterrows():
            # Regularizar la geometría
            regularized_geometry = row['geometry'].simplify(simp_fac, preserve_topology=topol)
            
            # Convertir a Polygon si es necesario
            if not isinstance(regularized_geometry, Polygon):
                if regularized_geometry.geom_type == 'MultiPolygon':
                    regularized_geometry = unary_union(regularized_geometry)
                elif regularized_geometry.geom_type == 'GeometryCollection':
                    regularized_geometry = max(
                        regularized_geometry, key=lambda geom: geom.area
                    )
            
            # Agregar la geometría regularizada a la lista
            regularized_geometries.append(regularized_geometry)
        
        # Crear un nuevo GeoDataFrame para la clase regularizada
        gdf_regularized = gpd.GeoDataFrame({'geometry': regularized_geometries}, crs=gdf.crs)
        
        # Definir la ruta de salida para el shapefile regularizado
        shp_out = os.path.join(output_dir, f'class_{class_id}_regularized.shp')
        
        # Guardar el shapefile regularizado
        gdf_regularized.to_file(shp_out)

        print(f'Clase {class_id}: shapefile regularizado guardado en {shp_out}')

# Uso de la función
shp_regularize_multi_class('input_shapefile.shp', 'output_directory', simp_fac=0.01, topol=True)

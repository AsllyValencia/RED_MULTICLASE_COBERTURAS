
# Red U-Net para Segmentación Multi-Clase


Este repositorio contiene el código para un proyecto de segmentación de imágenes utilizando una red U-Net personalizada.
El objetivo del proyecto es segmentar imágenes RGB de alta definición en cinco clases distintas: vegetación, cuerpos de agua, 
construcción, vías y otros. La estructura del proyecto incluye la implementación de funciones específicas para el entrenamiento 
del modelo, la generación y el procesamiento de shapefiles, así como la combinación de funciones de pérdida para mejorar el rendimiento 
de la segmentación.

## Contenido

- **`Weights/`**: Carpeta que contiene archivos relacionados con el manejo de los pesos del modelo, incluyendo el guardado y la carga de los mismos.
- **`DataProcessing/`**: Código y scripts para la preprocesamiento de datos y la creación de la base de datos personalizada.
- **`Model/`**: Implementación de la red U-Net y otras funciones relacionadas con el entrenamiento y la evaluación del modelo.
- **`ShapefileProcessing/`**: Funciones para generar y regularizar shapefiles que representan las diferentes clases de segmentación.
- **`Train.py`**: Script principal para entrenar el modelo U-Net desde cero utilizando la base de datos creada.
- **`Evaluate.py`**: Script para evaluar el rendimiento del modelo y generar métricas de evaluación.
- **`requirements.txt`**: Archivo que enumera los paquetes y versiones necesarios para ejecutar el proyecto.

## Requisitos

Para ejecutar el proyecto, necesitarás tener instalados los siguientes paquetes y dependencias:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Geopandas
- Shapely
- Otros paquetes según se especifique en `requirements.txt`

Instala todas las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

## Estimación de Ahorro

Este proyecto está diseñado para ahorrar tiempo en el procesamiento y análisis de grandes volúmenes de imágenes geoespaciales.
La automatización de la segmentación y la generación de shapefiles reduce significativamente el tiempo necesario para la clasificación manual.
Además, al combinar funciones de pérdida y ajustar el modelo específicamente para las cinco clases, el proyecto busca optimizar el rendimiento 
y mejorar la precisión de la segmentación, lo que resulta en una mayor eficiencia y precisión en el análisis de imágenes.

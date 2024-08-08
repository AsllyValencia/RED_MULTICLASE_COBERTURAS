import os,shutil

def delete_folders_with_prefix(directory, prefix):
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs.copy():
            if dir_name.startswith(prefix):
                dir_to_delete = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_to_delete)
                except:
                  pass

############# codigo modificado############

import os
import shutil

def delete_folders_with_prefix(directory, prefix):
    """
    Elimina las carpetas en el directorio especificado que comienzan con el prefijo dado.

    # Argumentos
        directory: Ruta al directorio principal donde se buscarán las carpetas.
        prefix: Prefijo que deben tener las carpetas a eliminar.
    """
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs.copy():
            if dir_name.startswith(prefix):
                dir_to_delete = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_to_delete)
                    print(f"Eliminada carpeta: {dir_to_delete}")
                except Exception as e:
                    print(f"No se pudo eliminar la carpeta {dir_to_delete}. Error: {e}")

# Ejemplo de uso
# delete_folders_with_prefix('./mi_directorio', 'prefijo_a_eliminar')

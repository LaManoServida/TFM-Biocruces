import os


def buscar_csv(ruta):
    """ Dada una ruta de una carpeta, devuelve una lista de archivos csv contenidos en ella """
    return [ruta + arch for arch in os.listdir(ruta) if arch.endswith('.csv')]

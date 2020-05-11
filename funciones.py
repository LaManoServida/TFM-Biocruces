import os


def buscar_csv(ruta):
    return [ruta + arch for arch in os.listdir(ruta) if arch.endswith('.csv')]

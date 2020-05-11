import pandas as pd
from funciones import buscar_csv

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/modificado/'
clave_principal = 'PATNO'

# POR CADA TABLA DE DATOS
for ruta_archivo in buscar_csv(ruta_carpeta):
    tabla = pd.read_csv(ruta_archivo, sep=',')

    # TODO: combinar tablas
    # datos = pd.merge(izq, der, on=clave_principal) ??

# guardar archivo único
# datos.to_csv(ruta_carpeta + 'fusionado.csv')

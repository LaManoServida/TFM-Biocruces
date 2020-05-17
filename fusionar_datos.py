import os
from functools import reduce
import pandas as pd
from funciones import buscar_csv, guardar_tabla

''' Fusiona todos los archivos csv de una carpeta en base a una clave principal común'''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/'
clave_principal = 'PATNO'
archivo_salida = 'fusionado.csv'

# cargar tablas
tablas = [pd.read_csv(ruta_arch, sep=',', float_precision='round_trip') for ruta_arch in buscar_csv(ruta_carpeta)]

datos = reduce(lambda t1, t2: pd.merge(t1, t2, on=clave_principal, how='outer'), tablas)  # ????

guardar_tabla(datos, os.path.join(ruta_carpeta, archivo_salida), '')

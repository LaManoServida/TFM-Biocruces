import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocesamiento.funciones import buscar_csv

''' Calcula la media del número de registros del mismo paciente en cada archivo '''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/3 - Fecha a timestamp'
ruta_grafico = 'Media pacientes repetidos - diagrama de barras.pdf'
clave_principal = 'PATNO'

# leer tablas
tablas = [pd.read_csv(ruta_arch, sep=',', float_precision='round_trip') for ruta_arch in buscar_csv(ruta_carpeta)]

# obtener los registros de cada archivo
registros = [tabla[clave_principal].values for tabla in tablas]

# número de registros de cada archivo
nums_registros = [len(registros_arch) for registros_arch in registros]

# número de pacientes únicos de cada archivo
nums_pacientes = [len(np.unique(registros_arch)) for registros_arch in registros]

# medias de repeticiones
medias_repeticiones = [nums_registros[i] / nums_pacientes[i] for i in range(len(registros))]

# generar diagrama de barras
etiquetas = [os.path.splitext(os.path.basename(r))[0] for r in buscar_csv(ruta_carpeta)]
plt.figure(figsize=(6, 7))
plt.bar(etiquetas, medias_repeticiones, color='royalblue')

# marcas horizontales
axes = plt.gca()
axes.yaxis.grid()

# etiquetas eje x
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

# título
plt.title('Media de nº de registros del mismo paciente')

# guardar
plt.tight_layout()
plt.savefig(ruta_grafico)

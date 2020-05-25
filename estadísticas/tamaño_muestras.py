import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocesamiento.funciones import buscar_csv

''' Calcula los tamaños de muestra de los archivos de una carpeta '''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/3 - Cambiar nombres de variables'
ruta_grafico = 'Tamaño de muestras - diagrama de barras.png'
clave_principal = 'PATNO'

# leer tablas
tablas = [pd.read_csv(ruta_arch, sep=',', float_precision='round_trip') for ruta_arch in buscar_csv(ruta_carpeta)]

# obtener solo el número de pacientes únicos de cada archivo
nums_pacientes = [len(np.unique(tabla[clave_principal].values)) for tabla in tablas]
print(nums_pacientes)

# generar diagrama de barras
etiquetas = [os.path.splitext(r)[0] for r in os.listdir(ruta_carpeta)]
print(etiquetas)
plt.figure(figsize=(6, 7))
plt.bar(etiquetas, nums_pacientes, color='forestgreen')

# marcas horizontales
axes = plt.gca()
axes.yaxis.grid()

# etiquetas eje x
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

# título
plt.title('Número de pacientes únicos por cada archivo')

# guardar
plt.tight_layout()
plt.savefig(ruta_grafico)

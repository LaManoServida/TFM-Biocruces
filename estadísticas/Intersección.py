import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocesamiento.funciones import buscar_csv

''' Calcula los porcentajes de intersección de pacientes entre varios archivos '''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/3 - Cambiar nombres de variables'
ruta_grafico = 'Intersección tablas - mapa de calor.png'
clave_principal = 'PATNO'

# leer tablas
tablas = [pd.read_csv(ruta_arch, sep=',', float_precision='round_trip') for ruta_arch in buscar_csv(ruta_carpeta)]
num_tablas = len(tablas)

# obtener solo los pacientes únicos de cada archivo
listas_pacientes: List = [np.unique(tabla[clave_principal].values).tolist() for tabla in tablas]

# calcular la matriz de intersección de pacientes
matriz = np.zeros((num_tablas, num_tablas), dtype=int)
for i in range(num_tablas):
    for j in range(num_tablas):
        num_pacientes_i = len(listas_pacientes[i])
        num_interseccion = len(set(listas_pacientes[i]) & set(listas_pacientes[j]))
        matriz[i, j] = round(num_interseccion / num_pacientes_i * 100)

print(matriz)

# generar mapa de calor
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(matriz, cmap='hot')

# número de marcas en cada eje
ax.set_xticks(np.arange(num_tablas))
ax.set_yticks(np.arange(num_tablas))

# etiquetas de cada eje
etiquetas = [os.path.splitext(r)[0] for r in os.listdir(ruta_carpeta)]
ax.xaxis.tick_top()
ax.set_xticklabels(etiquetas)
ax.set_yticklabels(etiquetas)

# configurar las etiquetas
plt.setp(ax.get_xticklabels(), rotation=35, ha='left', rotation_mode='anchor')

# mostrar texto en cada celda
for i in range(num_tablas):
    for j in range(num_tablas):
        ax.text(j, i, str(matriz[i, j]) + '%' if i != j else '-',
                ha='center', va='center', color='gray')

# título
ax.set_title('Porcentaje de pacientes de A contenidos en B', y=-0.1)

# guardar
fig.tight_layout()
plt.savefig(ruta_grafico)

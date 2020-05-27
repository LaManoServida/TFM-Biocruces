import os
from typing import List
import numpy as np
import pandas as pd
from preprocesamiento.funciones import buscar_csv
from estadísticas.generar_mapa_de_calor import mapa_de_calor

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
etiquetas = [os.path.splitext(r)[0] for r in os.listdir(ruta_carpeta)]
mapa_de_calor(matriz=matriz,
              ruta=ruta_grafico,
              titulo='Porcentaje de pacientes de A contenidos en B',
              etiquetas_x=etiquetas,
              etiquetas_y=etiquetas,
              sufijo='%',
              tamano_fig=(10, 10),
              rango_color=(0, 100)
              )

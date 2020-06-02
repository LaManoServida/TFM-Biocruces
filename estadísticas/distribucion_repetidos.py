import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocesamiento.funciones import buscar_csv

''' Calcula la distribución del número de registros del mismo paciente de cada archivo '''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/5 - Dividir entre tipos de pacientes (README)/Sanos con withdrew'
clave_principal = 'PATNO'
nombre_grafico = 'Distribución nº de consultas de cada paciente'

# por cada tabla
for ruta_arch in buscar_csv(ruta_carpeta):
    tabla = pd.read_csv(ruta_arch, sep=',', float_precision='round_trip')

    # nombre del gráfico
    nombre_archivo = os.path.splitext(os.path.basename(ruta_arch))[0]
    titulo = f'{nombre_grafico}\n{nombre_archivo}'
    ruta_grafico = os.path.join(ruta_carpeta, f'{nombre_grafico} - {nombre_archivo}.png')

    # obtener los pacientes en formato array
    registros = tabla[clave_principal].values

    # distribución del número repeticiones de cada paciente
    _, distribucion = np.unique(registros, return_counts=True)

    # generar diagrama de barras
    plt.hist(distribucion, density=True, color='royalblue')
    plt.title(titulo)

    # guardar
    plt.savefig(ruta_grafico)
    plt.clf()

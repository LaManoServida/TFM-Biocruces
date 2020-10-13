import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocesamiento.funciones import buscar_csv

''' Calcula la distribución del tiempo transcurrido entre mediciones baseline (> 1) de distintos archivos '''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/5 - Elegir BL entre BL y SC'
clave_principal = 'PATNO'
clave_fecha = 'INFODT'
nombre_grafico = 'Histograma tiempo entre baseline'

# coger las tablas que tiene fecha
tablas_con_fecha = []
for ruta_arch in buscar_csv(ruta_carpeta):
    tabla = pd.read_csv(ruta_arch, sep=',', float_precision='round_trip')
    if clave_fecha in tabla.columns:
        tablas_con_fecha.append(tabla[[clave_principal, clave_fecha]])

# pacientes sin duplicados
pacientes = list(set([paciente for tabla in tablas_con_fecha for paciente in tabla[clave_principal].to_list()]))

# crear un diccionario con estos pacientes y fechas como listas vacías
distribuciones = {p: [] for p in pacientes}

# rellenar las listas
for tabla in tablas_con_fecha:  # por cada archivo
    for i in range(len(tabla)):  # por cada registro
        distribuciones[tabla.loc[i][clave_principal]].append(tabla.loc[i][clave_fecha])

# calcular la media de las diferencias entre elementos de las listas, en meses
distribuciones = {c: np.mean(np.diff(np.sort(v))) / (3600 * 24 * 30) for c, v in distribuciones.items() if len(v) > 1}
valores = list(distribuciones.values())

# generar diagrama de barras
plt.figure(figsize=(8, 5))
plt.hist(valores, color='royalblue', range=(0, 24), bins=11)
plt.xlabel('Meses')
plt.ylabel('Pacientes')
plt.axvline(np.median(valores), color='k', linestyle='dashed', linewidth=2)
plt.text(np.median(valores) * 1.1, 520, 'Mediana: {:.2f}'.format(np.median(valores)))

# guardar
plt.savefig(os.path.join(ruta_carpeta, nombre_grafico + '.pdf'))
plt.show()
plt.clf()

from os.path import join
import pandas as pd
from preprocesamiento.funciones import buscar_csv, guardar_tabla

''' Filtra filas de varios archivos según los pacientes que aparezcan en una tabla de referencia'''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/6 - Dividir entre tipos de pacientes/RBD'
ruta_archivo_referencia = join(ruta_carpeta, 'Patient_Status_sin_duplicados_filtrado_por_valores.csv')
clave_principal = 'PATNO'

# leer tabla de referencia
tabla_ref = pd.read_csv(ruta_archivo_referencia, sep=',', float_precision='round_trip')

# leer cada tabla y filtrar según los pacientes que aparecen en el archivo de referencia
for ruta_arch in buscar_csv(ruta_carpeta):
    tabla = pd.read_csv(ruta_arch, sep=',', float_precision='round_trip')

    # guardar solo los pacientes que aparecen en el de referencia
    tabla = tabla[tabla[clave_principal].isin(tabla_ref[clave_principal])]

    # guardar tabla
    guardar_tabla(tabla, ruta_arch, '_filtrado_segun_ref')

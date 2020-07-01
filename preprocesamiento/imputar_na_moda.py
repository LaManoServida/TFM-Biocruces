import pandas as pd
from preprocesamiento.funciones import guardar_tabla

''' Reemplaza los valores perdidos por la moda de la columna '''

# PARÁMETROS
ruta = 'D:/Dropbox/UNI/TFM/datos/10 - Quitar variables y pacientes/IDIOPATHIC PD_con_variables_derivadas.csv'
excluir_variables = ['PATNO']

# leer tabla
tabla = pd.read_csv(ruta, sep=',', float_precision='round_trip')

# imputar la moda
tablita = tabla[tabla.columns.difference(excluir_variables)]
tabla[tabla.columns.difference(excluir_variables)] = tablita.fillna(tablita.mode().loc[0])

guardar_tabla(tabla, ruta, '_imputado')

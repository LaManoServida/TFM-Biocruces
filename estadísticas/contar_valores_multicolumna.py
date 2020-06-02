import os
import pandas as pd

''' Genera un informe que indica cuántos valores hay por cada valor posible de cada variable '''

# PARÁMETROS
ruta = 'D:/Dropbox/UNI/TFM/datos/4 - Convertir fecha a timestamp/Patient_Status.csv'
nombre_informe = 'Cuenta de valores multicolumna.txt'
clave_principal = 'PATNO'

# leer tabla
tabla = pd.read_csv(ruta, sep=',', float_precision='round_trip')

# crear archivo del informe
f = open(os.path.join(os.path.dirname(ruta), nombre_informe), mode='w', encoding='UTF-8')

# agrupar por cada valor posible de cada variable (quitando la calve principal)
grupos = tabla.groupby([c for c in tabla.columns.tolist() if c != clave_principal]).size()

# escribir en el archivo
f.write(grupos.to_string())
f.close()

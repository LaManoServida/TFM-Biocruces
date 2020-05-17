import pandas as pd
from funciones import guardar_tabla

''' Reemplaza los valores perdidos por la media de la fila o columna '''

# PARÁMETROS
ruta = 'D:/Dropbox/UNI/TFM/datos/UPDRS 3.csv'
subset_col = slice(3, -1)  # columnas a tener en cuenta [desde, hasta)
media_de_fila = True  # imputar valores por la media de la fila

# leer tabla
tabla_entera = pd.read_csv(ruta, sep=',', float_precision='round_trip')
tabla = tabla_entera.iloc[:, subset_col]  # solo las columnas seleccionadas

# borrar filas que solo tienen valores perdidos
num_perdidos = tabla.T.isna().sum().values  # array de número valores perdidos de cada registro
tabla = tabla[num_perdidos < len(tabla.columns)]

# imputar la media
if media_de_fila:
    nueva = tabla.T.fillna(round(tabla.mean(axis=1))).T
else:
    nueva = tabla.fillna(round(tabla.mean()))

# reconstruir la tabla entera
tabla_entera.iloc[:, subset_col] = nueva

guardar_tabla(tabla_entera, ruta, '_sin_na')

import os
import numpy as np
import pandas as pd

''' Calcula estadísticas referentes a los valores perdidos de un archivo '''

# PARÁMETROS
ruta_archivo = 'D:/Dropbox/UNI/TFM/datos/8 - Crear variables derivadas/HC_con_variables_derivadas.csv'
nombre_informe = 'Informe valores perdidos.txt'
excluir_variables = []

# crear archivo del informe
f = open(os.path.join(os.path.dirname(ruta_archivo), nombre_informe), mode='w', encoding='UTF-8')

# leer tabla
tabla = pd.read_csv(ruta_archivo, sep=',', float_precision='round_trip')

# excluir las variables indicadas en parámetros
if excluir_variables and isinstance(excluir_variables, list):
    tabla.drop(excluir_variables, 1, inplace=True)

# numero de filas
num_filas = len(tabla)

# numero de filas con algún valor perdido
filas_con_na = np.count_nonzero(tabla.T.isna().sum().values)

# escribir la cabecera
f.write(
    f'VALORES PERDIDOS DE {ruta_archivo}\n'
    f'Número de filas: {num_filas}\n'
    f'Filas con algún valor perdido: {filas_con_na} ({round(filas_con_na / num_filas * 100, 1)}%)\n\n'
    f'{"VARIABLE":17s}DATOS_CON_VALOR\n'
)

# rellenarlo por cada variable
for variable in tabla.columns:
    # número de valores perdidos de esa variable
    datos_no_perdidos = num_filas - tabla[variable].isna().sum()

    # porcentaje
    porcent_na_variable = round(datos_no_perdidos / num_filas * 100, 1)

    # escribir en el informe
    f.write(f'{str(variable):17s}{str(datos_no_perdidos):7s}({porcent_na_variable}%)\n')

f.close()

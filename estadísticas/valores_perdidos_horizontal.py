import os
import pandas as pd

''' Calcula estadísticas referentes a los valores perdidos de los pacientes de un archivo '''

# PARÁMETROS
ruta_archivo = 'D:/Dropbox/UNI/TFM/datos/12 - Imputar por moda/HC_con_variables_derivadas_imputado.csv'
nombre_informe = 'Informe valores perdidos (horizontal).txt'
excluir_variables = ['age_dis_onset']

# crear archivo del informe
f = open(os.path.join(os.path.dirname(ruta_archivo), nombre_informe), mode='w', encoding='UTF-8')

# leer tabla
tabla = pd.read_csv(ruta_archivo, sep=',', float_precision='round_trip')

# excluir las variables indicadas en parámetros
if excluir_variables and isinstance(excluir_variables, list):
    tabla.drop(excluir_variables, 1, inplace=True)

# número de filas
num_filas = len(tabla)

# número de columnas
num_columnas = len(tabla.columns)

# escribir la cabecera
f.write(f'VALORES PERDIDOS DE {ruta_archivo}\n'
        f'Número de filas: {num_filas}\n'
        f'Número de columnas: {num_columnas}\n\n')

# número de valores perdidos por cada paciente
num_na_filas = tabla.T.isna().sum()
num_na_filas_porc = [round(pac / num_columnas * 100, 2) for pac in num_na_filas]

# número de pacientes con i valores perdidos
i_na_pacientes = [(num_na_filas == i).sum() for i in range(num_columnas)]
i_na_pacientes_porc = [round(pac / num_filas * 100, 2) for pac in i_na_pacientes]

# escribir por cada número de valores perdidos != 0 pacientes
f.write(f'{"#NA":5s}#Pacientes\n')

for i in range(num_columnas):
    if i_na_pacientes[i] != 0:
        f.write(f'{str(i) + ":":5s}{str(i_na_pacientes[i]) + ",":5s}{str(i_na_pacientes_porc[i]) + "%":>6s}\n')

# escribir por cada paciente el número de valores perdidos que tiene
f.write(f'\n{"PATNO":7s}#NA\n')

for i in range(num_filas):
    f.write(f'{str(tabla.loc[i]["PATNO"]) + ":":7s}{str(num_na_filas[i]) + ",":5s}{num_na_filas_porc[i]:.2f}%\n')

# # rellenarlo por cada variable
# for variable in tabla.columns:
#     # número de valores perdidos de esa variable
#     datos_no_perdidos = num_filas - tabla[variable].isna().sum()
#
#     # porcentaje
#     porcent_na_variable = round(datos_no_perdidos / num_filas * 100, 1)
#
#     # escribir en el informe
#     f.write(f'{str(variable):17s}{str(datos_no_perdidos):7s}({porcent_na_variable}%)\n')

f.close()

import os
import pandas as pd

''' Calcula estadísticas referentes a los valores perdidos de un archivo '''

# PARÁMETROS
ruta_archivo = 'D:/Dropbox/UNI/TFM/datos/7 - Juntar en un solo archivo/IDIOPATHIC PD/fusionado.csv'
nombre_informe = 'Informe valores perdidos.txt'
clave_principal = 'PATNO'

# leer tabla
tabla = pd.read_csv(ruta_archivo, sep=',', float_precision='round_trip')

# numero de filas
num_filas = len(tabla)

# crear archivo del informe
f = open(os.path.join(os.path.dirname(ruta_archivo), nombre_informe), mode='w', encoding='UTF-8')

# escribir la cabecera
f.write(
    f'VALORES PERDIDOS DE {ruta_archivo}\n'
    f'Número de filas: {num_filas}\n\n'
)

# rellenarlo por cada variable
for variable in tabla.columns:
    # número de valores perdidos de esa variable
    na_variable = tabla[variable].isna().sum()

    # porcentaje
    porcent_na_variable = round(na_variable / num_filas * 100, 1)

    # escribir en el informe
    f.write(
        f'{str(variable):17s}'
        f'Datos: {str(num_filas - na_variable):7s}'
        f'NA: {str(na_variable):5s}({porcent_na_variable}%)\n'
    )

f.close()

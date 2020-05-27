import os
import numpy as np
import pandas as pd
from preprocesamiento.funciones import buscar_csv
from estadísticas.generar_mapa_de_calor import mapa_de_calor

''' Calcula estadísticas referentes a los valores perdidos de cada archivo '''

# PARÁMETROS
ruta_carpeta_datos = 'D:/Dropbox/UNI/TFM/datos/3 - Cambiar nombres de variables'
ruta_resultados = 'Valores perdidos - resultados'
nombre_informe = 'Informe valores perdidos.txt'
nombre_mapa_de_calor = 'Mapa de calor.png'
clave_principal = 'PATNO'

# crear carpeta para resultados
os.mkdir(ruta_resultados) if not os.path.exists(ruta_resultados) else None

# crear archivo del informe
f = open(os.path.join(ruta_resultados, nombre_informe), mode='w', encoding='UTF-8')

# rellenarlo por cada tabla
for ruta_archivo in buscar_csv(ruta_carpeta_datos):
    nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]

    # leer tabla
    tabla = pd.read_csv(ruta_archivo, sep=',', float_precision='round_trip')
    del tabla[clave_principal]  # quitar clave principal porque no tiene valores perdidos

    # número de valores perdidos por cada columna
    na_columnas = tabla.isna().sum()

    # número total de valores perdidos
    total_na = na_columnas.sum()

    # porcentaje de valores perdidos
    porcent_total_na = round(total_na / np.prod(tabla.shape) * 100, 1)

    # número de filas con valores perdidos
    filas_con_na = np.count_nonzero(tabla.T.isna().sum().values)

    # porcentaje de filas con valores perdidos
    porcent_filas_con_na = round(filas_con_na / len(tabla) * 100, 1)

    # incidencia de los valores perdidos en cada variable
    prop_na_columnas = round(na_columnas / total_na * 100, 1)

    # escribir en el informe
    f.write(
        f'--- {nombre_archivo} ---\n'
        f'Dimensiones: {tabla.shape}\n'
        f'Valores perdidos: {total_na}, un {porcent_total_na}%\n'
        f'Filas con valores perdidos: {filas_con_na}, un {porcent_filas_con_na}%\n'
        f'Incidencia por variable:\n'
    )
    for variable, valor in zip(prop_na_columnas.keys().values, prop_na_columnas.values):
        f.write(f'{variable:17s}{valor}%\n')

    # separador entre tablas
    f.write('\n' * 3)

    # generar mapas de calor mostrando la correlación entre variables para valores perdidos
    mapa_de_calor(matriz=tabla.isna().corr().values.round(1),
                  ruta=os.path.join(ruta_resultados, f'{nombre_archivo} - {nombre_mapa_de_calor}'),
                  titulo=f'Correlación entre variables para valores perdidos\n{nombre_archivo}',
                  etiquetas_x=tabla.columns.to_list(),
                  etiquetas_y=tabla.columns.to_list(),
                  paleta='bwr',
                  rango_color=(-1, 1)
                  )

f.close()

import os
from functools import reduce
import pandas as pd
from preprocesamiento.funciones import buscar_csv, guardar_tabla

''' Fusiona todos los archivos csv de una carpeta en base a una clave principal común '''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/'
clave_principal = 'PATNO'
tabla_principal = 'Principal_sin_duplicados.csv'
archivo_salida = 'fusionado.csv'


def combinar_tablas(t1, t2):
    """ Combina dos tablas, dando preferencia a la segunda si hay columnas duplicadas """
    # columnas que tienen en común
    cols_comun = t1.columns.intersection(t2.columns)

    # a parte de la clave principal, solo puede haber una más en común
    if clave_principal not in cols_comun or len(cols_comun) > 2:
        print('No ha sido posible combinar la tabla {} con {}'.format(t1.columns.values, t2.columns.values))
        return t1

    # combinar tablas según la clave principal
    sufijos_dup = ('_dup1', '_dup2')  # sufijos para posibles columnas duplicadas
    fusionada = pd.merge(t1, t2, on=clave_principal, how='outer', suffixes=sufijos_dup)

    # pero si tiene otra más en común, pueden quedar columnas duplicadas
    if len(cols_comun) == 2:
        la_otra_comun = cols_comun.drop(clave_principal).values[0]
        cols_duplicadas = [la_otra_comun + suf for suf in sufijos_dup]  # nombres en la tabla de las columnas duplicadas

        # combinar las dos columnas dando preferencia a la última y guardarlo en une nueva
        fusionada[la_otra_comun] = fusionada[cols_duplicadas[1]].combine_first(fusionada[cols_duplicadas[0]])

        # borrar estas dos sobrantes
        fusionada.drop(cols_duplicadas, 1, inplace=True)

    return fusionada


# leer archivos
rutas_arch = buscar_csv(ruta_carpeta)

# poner la tabla principal al comienzo de la lista
rutas_arch.insert(0, rutas_arch.pop(rutas_arch.index(os.path.join(ruta_carpeta, tabla_principal))))

# cargar tablas
tablas = [pd.read_csv(ruta_arch, sep=',', float_precision='round_trip') for ruta_arch in rutas_arch]

# combinarlas todas aglomerativamente
datos = reduce(lambda t1, t2: combinar_tablas(t1, t2), tablas)

guardar_tabla(datos, os.path.join(ruta_carpeta, archivo_salida), '')

import pandas as pd
from funciones import buscar_csv

''' Descarta los registros diferentes del mismo paciente, cogiendo el más completo o reciente '''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/modificado/'
clave_principal = 'PATNO'

# POR CADA TABLA DE DATOS
for ruta_archivo in buscar_csv(ruta_carpeta):
    tabla = pd.read_csv(ruta_archivo, sep=',')

    # TODO
    # si existen varios registros del mismo paciente:
    #     si hay un registro con menos valores perdidos:
    #         descartar los demás
    #     si no (todos tienen el mismo número de valores perdidos):
    #         si la tabla tiene una columna de fecha (porque no está ordenada cronológicamente):
    #             descartar los registros anteriores de cada paciente
    #         para el resto (y para las anteriores si todavía hay duplicidad con la misma fecha)
    #         coger el último registro de cada paciente 

    # guardar cambios sobrescribiendo
    tabla.to_csv(ruta_archivo, index=False)

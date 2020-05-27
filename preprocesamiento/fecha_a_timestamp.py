import time
from datetime import datetime as dt
import pandas as pd
from preprocesamiento.funciones import guardar_tabla

''' Convierte la fecha en ISO 8601 de una columna a UNIX timestamp '''

# PARÁMETROS
ruta = 'D:/Dropbox/UNI/TFM/datos/4 - Convertir fecha a timestamp/University_of_Pennsylvania_Smell_ID_Test.csv'
variable_fecha = 'FECHA'

# leer tabla
tabla = pd.read_csv(ruta, sep=',', float_precision='round_trip')


# funcnión para convertir fecha
def convertir_fecha(fecha):
    if pd.isna(fecha):
        return fecha
    if len(fecha) > 7:  # si tiene hora
        return int(time.mktime(dt.strptime(fecha, '%m/%Y %H:%M:%S').timetuple()))
    else:
        return int(time.mktime(dt.strptime(fecha, '%m/%Y').timetuple()))


# convertir a UNIX
tabla[variable_fecha] = tabla[variable_fecha].map(convertir_fecha)

guardar_tabla(tabla, ruta, '_fecha_timestamp')

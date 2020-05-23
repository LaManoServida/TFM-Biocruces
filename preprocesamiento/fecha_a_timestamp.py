import time
import ciso8601
import pandas as pd
from preprocesamiento.funciones import guardar_tabla

''' Convierte la fecha en ISO 8601 de una columna a UNIX timestamp '''

# PARÁMETROS
ruta = 'D:/Dropbox/UNI/TFM/datos/UPDRS 3.csv'
variable_fecha = 'FECHA'

# leer tabla
tabla = pd.read_csv(ruta, sep=',', float_precision='round_trip')

# convertir a UNIX
tabla[variable_fecha] = tabla[variable_fecha].map(
    lambda f: int(time.mktime(ciso8601.parse_datetime(f).timetuple()))
)

guardar_tabla(tabla, ruta, '_fecha_timestamp')

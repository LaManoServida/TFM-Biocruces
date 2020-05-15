import os
import time
import ciso8601
import pandas as pd

''' Convierte la fecha en ISO 8601 de una columna a UNIX timestamp '''

# PARÁMETROS
ruta = 'D:/Dropbox/UNI/TFM/datos/UPDRS 3.csv'
variable_fecha = 'FECHA'

# leer tabla
tabla = pd.read_csv(ruta, sep=',')

# convertir a UNIX
tabla[variable_fecha] = tabla[variable_fecha].map(
    lambda f: int(time.mktime(ciso8601.parse_datetime(f).timetuple()))
)

# guardar
ruta_split = os.path.splitext(ruta)
tabla.to_csv(ruta_split[0] + '_fecha_timestamp' + ruta_split[1], index=False)

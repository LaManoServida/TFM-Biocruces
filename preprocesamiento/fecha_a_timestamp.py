import time
from dateutil.parser import parse
import calendar
import locale
import pandas as pd
from preprocesamiento.funciones import buscar_csv, guardar_tabla

''' Convierte la columna de fecha a timestamp de todos los archivos csv de una carpeta '''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/3 - Fecha a timestamp/'
variable_fecha = 'INFODT'


# función para convertir fecha
def convertir_fecha(fecha):
    if pd.isna(fecha):
        return fecha

    # si el mes está escrito con letras cambiarlo a números
    if any(c.isalpha() for c in fecha):
        fecha = fecha.replace(fecha[0:3], meses[fecha[0:3]])

    return int(time.mktime(parse(fecha).timetuple()))


# nombres de meses en español
locale.setlocale(locale.LC_TIME, 'es')
meses = {calendar.month_abbr[i][:-1]: str(i) for i in range(1, 13)}

# convertir archivos
for ruta_archivo in buscar_csv(ruta_carpeta):
    tabla = pd.read_csv(ruta_archivo, sep=',', float_precision='round_trip')

    # si la tabla no tiene fecha, pasar
    if variable_fecha not in tabla.columns:
        continue

    # convertir a UNIX timestamp
    tabla[variable_fecha] = tabla[variable_fecha].map(convertir_fecha)

    guardar_tabla(tabla, ruta_archivo, '_fecha_timestamp')

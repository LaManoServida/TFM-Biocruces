import numpy as np
import pandas as pd
from preprocesamiento.funciones import guardar_tabla

''' Filtra filas de la tabla según los valores que tomen una o varias columnas '''

# PARÁMETROS
ruta_archivo = 'D:/Dropbox/UNI/TFM/datos/5 - Dividir entre tipos de pacientes (README)/Enfermos con withdrew/' \
               'Patient_Status.csv'
filtros_columna_valores = [('RECRUITMENT_CAT', ['PD']),  # lista de tuplas: (nombre de columna, lista de valores)
                           ('ENROLL_STATUS', ['Enrolled', 'Complete', 'Withdrew']),
                           ('DESCRP_CAT', ['IDIOP'])]

# leer tabla
tabla = pd.read_csv(ruta_archivo, sep=',', float_precision='round_trip')

# acumular cada condición de filtrado
mascara_acumulativa = np.full(len(tabla), True)  # inicializar máscara
for filtro in filtros_columna_valores:
    mascara_acumulativa &= tabla[filtro[0]].isin(filtro[1])  # construir y añadir submáscara

# aplicar máscara
tabla = tabla[mascara_acumulativa]

# guardar tabla
guardar_tabla(tabla, ruta_archivo, '_filtrado_por_valores')

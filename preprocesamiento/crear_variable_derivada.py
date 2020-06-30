from datetime import datetime as dt
import numpy as np
import pandas as pd
from preprocesamiento.funciones import guardar_tabla

''' Genera y añade una variable nueva, derivada a partir de otras de la tabla '''

# PARÁMETROS
ruta_archivo = 'D:/Dropbox/UNI/TFM/datos/7 - Juntar en un solo archivo/HC.csv'


def edad(fila):
    """ Calcula la edad a partir de la fecha de actual y la de nacimiento """
    if pd.isna(fila['INFODT']) or pd.isna(fila['BIRTHDT']):  # si alguna de las variables no está disponible
        return np.nan

    return round(
        (dt.fromtimestamp(int(fila['INFODT'])) - dt(year=int(fila['BIRTHDT']), month=1, day=1)).days / 365.25,
        2)


def edad_de_diagnostico(fila):
    """ Calcula la edad del diagnóstico a partir de la fecha de diagnóstico y la de nacimiento """
    if pd.isna(fila['PDDXDT']) or pd.isna(fila['BIRTHDT']):  # si alguna de las variables no está disponible
        return np.nan

    return round(
        (dt.fromtimestamp(int(fila['PDDXDT'])) - dt(year=int(fila['BIRTHDT']), month=1, day=1)).days / 365.25,
        2)


def flag_educacion(fila):
    """ Calcula si el número de años de educación es menor o igual a 12 """
    if pd.isna(fila['EDUCYRS']):
        return np.nan

    return int(fila['EDUCYRS'] <= 12)


def scopa_preguntas_sexo():
    """ Calcula la puntuación de las preguntas de sexo, independientemente de qué sexo sea """
    dicts = [{'var_hombre': 'SCAU22', 'var_mujer': 'SCAU24'},
             {'var_hombre': 'SCAU23', 'var_mujer': 'SCAU25'}]
    return [tabla.apply(scopa_preguntas_sexo_aux, **variables, axis=1) for variables in dicts]


def scopa_preguntas_sexo_aux(fila, var_hombre, var_mujer):
    if pd.isna(fila[var_hombre]) and pd.isna(fila[var_mujer]):
        return np.nan

    return fila[var_hombre] if pd.notna(fila[var_hombre]) else fila[var_mujer]


def benton_por_pares(fila, v1, v2):
    """ Calcula la puntuación de los pares de preguntas del test Benton """
    if pd.isna(fila[v1]) and pd.isna(fila[v2]):
        return np.nan

    return fila[v1] if pd.notna(fila[v1]) else fila[v2]


# leer tabla
tabla = pd.read_csv(ruta_archivo, sep=',', float_precision='round_trip')

# añadir variables derivadas
tabla['age'] = tabla.apply(edad, axis=1)
tabla['age_dis_onset'] = tabla.apply(edad_de_diagnostico, axis=1)
tabla['flag_edu'] = tabla.apply(flag_educacion, axis=1)
tabla['SCAUSEX1'], tabla['SCAUSEX2'] = scopa_preguntas_sexo()

for i in range(1, 16):
    tabla[f'BJLOTPAR{i}'] = tabla.apply(benton_por_pares, v1=f'BJLOT{2 * i - 1}', v2=f'BJLOT{2 * i}', axis=1)

guardar_tabla(tabla, ruta_archivo, '_con_variables_derivadas')

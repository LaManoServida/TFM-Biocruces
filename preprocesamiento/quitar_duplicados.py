import pandas as pd
from preprocesamiento.funciones import buscar_csv, guardar_tabla

''' Descarta los registros diferentes del mismo paciente, cogiendo el más completo y/o antiguo '''

# PARÁMETROS
ruta_carpeta = 'D:/Dropbox/UNI/TFM/datos/5 - Elegir BL entre BL y SC/bongo'
clave_principal = 'PATNO'
clave_orden = 'EVENT_ID'
coger_menor = True
tener_en_cuenta_num_na = False  # si tener en cuenta o no la completitud del registro como criterio para su selección

# leer cada tabla y quitar los duplicados
for ruta_arch in buscar_csv(ruta_carpeta):
    tabla = pd.read_csv(ruta_arch, sep=',', float_precision='round_trip')

    # quitar filas totalmente duplicadas
    tabla.drop_duplicates(inplace=True)

    # si existen varios registros del mismo paciente:
    if not tabla[clave_principal].is_unique:
        # coger lista de grupos de registros que tienen el mismo paciente
        lista_rep = [grupo for _, grupo in tabla.groupby(clave_principal) if len(grupo) > 1]  # lista de DataFrames

        # quitar todos los duplicados de la tabla para luego meter un único registro por paciente
        tabla.drop_duplicates(subset=clave_principal, keep=False, inplace=True)

        for grupo in lista_rep:
            if tener_en_cuenta_num_na:
                # quedarse con los registros con menos valores perdidos
                num_perdidos = grupo.T.isna().sum().values  # array de número valores perdidos de cada registro
                grupo = grupo[num_perdidos == num_perdidos.min()]

            # de ellos elegir el que menor/mayor valor de la clave por la que se ordenan tenga
            if clave_orden in grupo:
                grupo = grupo.sort_values(clave_orden, ascending=coger_menor)  # se ordenan

            tabla = pd.concat((tabla, grupo.iloc[[0]]))  # añadir el primero

    guardar_tabla(tabla, ruta_arch, '_sin_duplicados')

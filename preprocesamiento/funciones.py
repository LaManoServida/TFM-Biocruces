import os


def buscar_csv(ruta_carpeta):
    """ Dada una ruta de una carpeta, devuelve las rutas de los archivos csv contenidos en ella """
    return [os.path.join(ruta_carpeta, arch) for arch in os.listdir(ruta_carpeta) if arch.endswith('.csv')]


def guardar_tabla(tabla, ruta, sufijo=''):
    """ Guarda la tabla en la ruta indicada """
    ruta_split = os.path.splitext(ruta)
    tabla.to_csv(ruta_split[0] + sufijo + ruta_split[1], index=False)


def aplicar_regex_a_lista(lista_variables, datos):
    """ Devuelve la lista de variables de los datos que coinciden con el patrón regex de cada variable de la lista """

    variables_encontradas = []
    for variable in lista_variables:
        variables_encontradas += datos.loc[:, datos.columns.str.match(variable)].columns.tolist()

    return variables_encontradas

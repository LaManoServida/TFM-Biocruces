import json

''' Genera una tabla de datos nueva, creada a partir de la suma (u otras operaciones) de algunas variables.
Además, se pueden definir valores diferentes para sumarlas, especificados en un archivo json '''


def agrupar_variables(datos_antiguo, ruta_sustitutos, ruta_sumatorios):
    # copiar la tabla de datos
    datos = datos_antiguo.copy()

    # abrir el diccionario de valores a sustituir
    with open(ruta_sustitutos) as sust:
        sustitutos = json.load(sust)

    # convertir claves de string a int
    sustitutos = {c: {int(c2): v2 for c2, v2 in v.items()} for c, v in sustitutos.items()}

    # sustituir los datos
    datos_sustituidos = datos.replace(sustitutos)

    # abrir el diccionario de agrupamiento de variables
    with open(ruta_sumatorios) as suma:
        sumatorios = json.load(suma)

    # calcular y añadir las variables hechas mediante sumatorios
    for var_derivada in sumatorios:
        datos[var_derivada] = sum([datos_sustituidos[sumando].values for sumando in sumatorios[var_derivada]])

    # [AD HOC] ponderar la suma de dos variables respecto a la cantidad de sumandos (media)
    # datos['Mean_TD'] /= len(sumatorios['Mean_TD'])
    # datos['Mean_RA'] /= len(sumatorios['Mean_RA'])

    # [AD HOC] añadir el TD_RA_ratio y TD_RA
    # datos['TD_RA_ratio'] = datos.apply(lambda fila: fila['Mean_TD'] / fila['Mean_RA'], axis=1)
    # datos['TD_RA'] = datos.apply(lambda fila: int(fila['TD_RA_RATIO'] < 0.8), axis=1)  # 0: TD, 1: RA

    # borrar las variables que han sido sumadas porque ya no se van a usar
    variables_a_borrar = list(set(  # sin duplicacaciones
        sum([sumatorios[var_derivada] for var_derivada in sumatorios], [])  # concatenar listas de sumandos
    ))
    datos.drop(variables_a_borrar, axis=1, inplace=True)

    return datos

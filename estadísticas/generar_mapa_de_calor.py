import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

''' Genera y guarda el mapa de calor de una matriz '''


def mapa_de_calor(matriz, ruta, titulo, etiquetas_x, etiquetas_y, sufijo='', no_diagonal=True, tamano_fig='auto',
                  paleta='hot', rango_color=(None, None)):
    # tamaño figura
    if tamano_fig == 'auto':
        tamano_fig = (6, 6) if max(matriz.shape) < 6 else np.asarray(matriz.shape) * 0.9

    # definir mapa de colores
    plt.set_cmap(paleta)
    plt.get_cmap().set_bad(color='gray')  # valores NaN en gris

    # si se ha especificado no pintar la diagonal
    tipo_datos_orig = matriz.dtype
    if no_diagonal:
        matriz = matriz.astype('float')  # para que pueda haber nan
        diagonal = range(min(matriz.shape))
        matriz[diagonal, diagonal] = np.nan

    # crear figura
    fig, ax = plt.subplots(figsize=tamano_fig)
    ax.imshow(matriz, vmin=rango_color[0], vmax=rango_color[1])

    # número de marcas en cada eje
    ax.set_xticks(np.arange(matriz.shape[1]))
    ax.set_yticks(np.arange(matriz.shape[0]))

    # etiquetas de cada eje
    ax.xaxis.tick_top()
    ax.set_xticklabels(etiquetas_x)
    ax.set_yticklabels(etiquetas_y)

    # configurar las etiquetas
    plt.setp(ax.get_xticklabels(), rotation=35, ha='left', rotation_mode='anchor')

    # mostrar texto en cada celda
    for i in range(matriz.shape[1]):
        for j in range(matriz.shape[0]):
            valor = matriz[i, j] if np.isnan(matriz[i, j]) else matriz[i, j].astype(tipo_datos_orig)  # no convertir nan
            ax.text(j, i, f'{valor}{sufijo}', ha='center', va='center', color='gray')

    # título
    ax.set_title(titulo, y=-0.1)

    # guardar
    fig.tight_layout()
    plt.savefig(ruta)
    plt.show()

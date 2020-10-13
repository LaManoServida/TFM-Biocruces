import os
import math
from itertools import compress
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from preprocesamiento.crear_variables_agrupadas import agrupar_variables
from preprocesamiento.funciones import aplicar_regex_a_lista
import matplotlib.pyplot as plt

# PARÁMETROS
ruta_datos = 'D:/Dropbox/UNI/TFM/datos/12 - Imputar por moda/IDIOPATHIC PD_con_variables_derivadas_imputado.csv'
ruta_datos_futuro = 'D:/Dropbox/UNI/TFM/datos/No supervisado/futuro.csv'
lista_negra_indiv = ['PATNO', 'ENROLL_STATUS', 'GENDER', 'HANDED', 'age', 'age_dis_onset']
# lista_negra_agrup = ['PATNO', 'GDSDROPD', 'GDSEMPTY', 'GDSAFRAD', 'GDSHAPPY', 'GDSHOME', 'GDSMEMRY', 'GDSALIVE',
#                      'GDSENRGY', 'GDSBETER', 'HVLTREC', 'HVLTFPRL', 'HVLTFPUN', 'PTCGBOTH',
#                      'MCATOT'] + lista_negra_indiv  # para intermedios 1 y 2
lista_negra_agrup = ['PATNO', 'HVLTREC', 'HVLTFPRL', 'HVLTFPUN', 'PTCGBOTH', 'MCATOT'] + lista_negra_indiv  # para total
variables_para_one_hot_regex = ['^ENROLL_STATUS$', '^GENDER$', '^HANDED$', '^SCAU'] + lista_negra_indiv
semilla = 0
hacer_variables_agrupadas = True
ruta_sustitutos = 'D:/Dropbox/UNI/TFM/datos/valores sustitutos.json'
ruta_sumatorios = 'D:/Dropbox/UNI/TFM/datos/sumatorios no supervisado total.json'

# elegir las variables dependiendo de si se hacen agrupamientos o no
lista_negra = lista_negra_indiv if not hacer_variables_agrupadas else lista_negra_agrup

# leer datos
datos = pd.read_csv(ruta_datos, sep=',', float_precision='round_trip')
patnos = datos['PATNO']

# aplicar regex a las variables para one hot
variables_para_one_hot = aplicar_regex_a_lista(variables_para_one_hot_regex, datos)

# quitar variables de la lista negra
datos.drop(lista_negra, axis=1, inplace=True)

# si hay que agrupar variables
if hacer_variables_agrupadas:
    datos = agrupar_variables(datos, ruta_sustitutos, ruta_sumatorios)

# quitar variables que no varían
datos.drop(datos.columns[datos.nunique().values == 1], axis=1, inplace=True)

# codificar en one hot las variables cualitativas no ordinales de más de 2 valores
# si ya no existe alguna variable, no tenerla en cuenta
variables_para_one_hot = list(compress(variables_para_one_hot, pd.Series(variables_para_one_hot).isin(datos.columns)))
cols_para_one_hot = datos[variables_para_one_hot]
one_hot = OneHotEncoder()
cat_onehot = pd.DataFrame(one_hot.fit_transform(cols_para_one_hot).toarray())
cat_onehot.columns = one_hot.get_feature_names(cols_para_one_hot.columns)  # cambiarles los nombres

# sustituir su versión anterior
datos.drop(cols_para_one_hot.columns, axis=1, inplace=True)
datos = datos.join(cat_onehot)

# escalar datos
datos = pd.DataFrame(MinMaxScaler().fit_transform(datos), index=datos.index, columns=datos.columns)

# TODO: selección de variables

# métricas
# https://stats.stackexchange.com/questions/99171/why-is-euclidean-distance-not-a-good-metric-in-high-dimensions/99191#99191


# visualizar dendrograma

# hacer clustering
params_silueta = pd.DataFrame(columns=['ncluster', 'affinity', 'link', 'silueta'])
params_silueta.silueta = pd.to_numeric(params_silueta.silueta)

for ncluster in range(2, 15):
    for link in ['ward', 'average', 'complete', 'single']:
        for affinity in ['manhattan', 'euclidean']:
            if link == 'ward' and affinity != 'euclidean':
                continue
            clustering = AgglomerativeClustering(n_clusters=ncluster, affinity=affinity, linkage=link).fit(datos)

            params_silueta.loc[-1] = [ncluster, affinity, link, silhouette_score(datos, clustering.labels_)]
            params_silueta.index += 1
            params_silueta = params_silueta.sort_index()

# mejor configuración
mejor_config = params_silueta.loc[params_silueta.silueta.idxmax()]
mejor_ncluster = mejor_config.ncluster
mejor_affinity = mejor_config.affinity
mejor_link = mejor_config.link
clustering = AgglomerativeClustering(n_clusters=mejor_ncluster, affinity=mejor_affinity, linkage=mejor_link).fit(datos)

print(f'Mejor con {mejor_ncluster} clusters, link {mejor_link} y affinity {mejor_affinity}: \
silueta {mejor_config.silueta}')

# pca
plt.figure(figsize=(6, 6))
pca = PCA(n_components=17)
datos_pca = pca.fit_transform(datos)
plt.scatter(datos_pca[:, 0], datos_pca[:, 1], s=5, c=clustering.labels_, cmap='prism')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
xvector = pca.components_[0] * max(datos_pca[:, 0])
yvector = pca.components_[1] * max(datos_pca[:, 1])

important_features = {datos.columns[i]: math.sqrt(xvector[i] ** 2 + yvector[i] ** 2) for i in range(len(datos.columns))}
important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
print('Atributos importantes:', important_features)

flechas_imprimir = [nombre for (num, nombre) in important_features][:5]
for i in range(len(datos.columns)):
    if datos.columns[i] in flechas_imprimir:
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.03, alpha=0.75)
        plt.text(xvector[i] * 1.2, yvector[i] * 1.2, datos.columns.to_list()[i], color='b', alpha=0.75)
plt.savefig('D:/Escritorio/awa.pdf')
plt.show()

# ahora leer datos futuro
datos_futuro = pd.read_csv(ruta_datos_futuro, sep=',', float_precision='round_trip')
# datos_futuro.drop(datos_futuro.columns.difference(['PATNO', 'EVENT_ID', 'INFODT', 'SUMA']), 1, inplace=True)

# pacientes de cada grupo
print('Diferencia de valores por cada cluster:')

for variable in datos_futuro.columns.drop(['EVENT_ID', 'PATNO', 'INFODT']):
    print('\n', variable)
    for i in range(mejor_ncluster):
        grupoi = patnos[clustering.labels_ == i]
        valores_grupoi = datos_futuro[variable][datos_futuro['PATNO'].isin(grupoi)]
        print(f'- Cluster {i}: {valores_grupoi.mean():.2f}')

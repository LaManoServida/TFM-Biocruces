import numpy as np
import os
from itertools import compress
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.tree import export_graphviz
import pickle
from matplotlib import pyplot as plt
from preprocesamiento.crear_variables_agrupadas import agrupar_variables
from preprocesamiento.funciones import aplicar_regex_a_lista

# PARÁMETROS
ruta_datos = 'D:/Dropbox/UNI/TFM/datos/14 - Juntar HC e IDIOPATHIC PD/HC + IDIOPATHIC PD.csv'
lista_negra_regex_indiv = ['^PATNO$', '^age_dis_onset$', '^NP1', '^NP2', '^NP3', '^NUPSOURC', '^DYSKPRES$', '^NHY$']
# lista_negra_regex_agrup = ['^PATNO$', '^age_dis_onset$', '^NP1', '^NP2', '^NP3', '^NUPSOURC', '^DYSKPRES$', '^NHY$',
#                            '^GDSDROPD$', '^GDSEMPTY$', '^GDSAFRAD$', '^GDSHAPPY$', '^GDSHOME$', '^GDSMEMRY$',
#                            '^GDSALIVE$', '^GDSENRGY$', '^GDSBETER$', '^HVLTREC$', '^HVLTFPRL$', '^HVLTFPUN$',
#                            '^PTCGBOTH$', '^MCATOT$']  # para intermedios 1 y 2
lista_negra_regex_agrup = ['^PATNO$', '^age_dis_onset$', '^NP1', '^NP2', '^NP3', '^NUPSOURC', '^DYSKPRES$', '^NHY$',
                           '^HVLTREC$', '^HVLTFPRL$', '^HVLTFPUN$', '^PTCGBOTH$', '^MCATOT$']  # para total
variables_para_one_hot_regex = ['^ENROLL_STATUS$', '^GENDER$', '^HANDED$', '^SCAU']
nombre_clase = 'Class'
semilla = 0
hacer_grid_search = False
hacer_variables_agrupadas = True
ruta_sustitutos = 'D:/Dropbox/UNI/TFM/datos/14 - Juntar HC e IDIOPATHIC PD/valores sustitutos.json'
ruta_sumatorios = 'D:/Dropbox/UNI/TFM/datos/14 - Juntar HC e IDIOPATHIC PD/sumatorios supervisado total.json'

# elegir las variables dependiendo de si se hacen agrupamientos o no
lista_negra_regex = lista_negra_regex_indiv if not hacer_variables_agrupadas else lista_negra_regex_agrup

# leer datos
datos = pd.read_csv(ruta_datos, sep=',', float_precision='round_trip')

# aplicar regex a las listas
lista_negra = aplicar_regex_a_lista(lista_negra_regex, datos)
variables_para_one_hot = aplicar_regex_a_lista(variables_para_one_hot_regex, datos)

# quitar variables de la lista negra
datos.drop(lista_negra, axis=1, inplace=True)

# si hay que agrupar variables
if hacer_variables_agrupadas:
    datos = agrupar_variables(datos, ruta_sustitutos, ruta_sumatorios)

# quitar variables que no varían
datos.drop(datos.columns[datos.nunique().values == 1], axis=1, inplace=True)

# codificar la variable ENROLL_STATUS con etiquetas numéricas
datos['ENROLL_STATUS'] = LabelEncoder().fit_transform(datos['ENROLL_STATUS'])

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

# separar atributos de clase
X = datos.drop(nombre_clase, axis=1)
y = datos[nombre_clase]

# separar train y test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=semilla, stratify=y.values)

# iterar pesos para la clase HC
px, py = [], []
for p in np.arange(0.5, 1.5, 0.001):
    print(p)

    # entrenar baseline: árbol de clasificación y sacar predicciones
    bl_y_pred = cross_val_predict(DecisionTreeClassifier(random_state=semilla), X, y)  # 5-fold CV

    if hacer_grid_search:
        # obtener el modelo random forest con mejores valores para los hiperparámetros con grid search
        valores_buscar_params = {'n_estimators': [100, 200],
                                 'criterion': ['gini', 'entropy'],
                                 'min_samples_split': [10, 11, 12, 13, 14],
                                 'min_samples_leaf': [1, 2, 3, 4, 5],
                                 'max_depth': [2, 3, 4, 5, 6, 7],
                                 'max_features': [None]  # , 'sqrt', 'log2'],
                                 # 'min_impurity_decrease': [0.1, 0.01, 0.001],
                                 # 'ccp_alpha': [0, 10, 100]
                                 }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=semilla), valores_buscar_params, n_jobs=4,
                                   verbose=3)
        grid_search.fit(X, y)  # probar y entrenar todas las combinaciones con 5-fold CV
        rf = grid_search.best_estimator_
    else:
        # crear random forest con los hiperparámetros especificados
        valores_params = {'n_estimators': 100,
                          'criterion': 'gini',
                          'min_samples_split': 12,
                          'min_samples_leaf': 4,
                          'max_depth': 4,
                          'max_features': None,
                          'min_impurity_decrease': 0.001,
                          'ccp_alpha': 0,
                          'class_weight': {'HC': 1, 'IDIOPATHIC PD': p}}
        rf = RandomForestClassifier(**valores_params, random_state=semilla)
        grid_search = None

    # entrenar y sacar predicciones y accuracy con CV
    y_pred = cross_val_predict(rf, X, y)  # 5-fold CV
    acc_rf_bueno = round(accuracy_score(y, y_pred), 4)

    # volver a entrenarlo pero ahora con todos los datos
    rf.fit(X, y)

    # guardarlo
    # pickle.dump(rf, open(f'Modelo {acc_rf_bueno}.pickle', 'wb'))

    # evaluar
    print('[BASELINE - ÁRBOL DE CLASIFICACIÓN]')
    print('Matriz de confusión:\n', confusion_matrix(y, bl_y_pred))
    print('Accuracy:', round(accuracy_score(y, bl_y_pred), 4), '\n')

    print('[RANDOM FOREST]')
    if hacer_grid_search:
        print(f'Grid search - Mejores parámetros encontrados: {grid_search.best_params_}')
    print('Matriz de confusión:\n', confusion_matrix(y, y_pred))
    print('Accuracy:', acc_rf_bueno, '\n')

    # # visualizar la importancia de las variables
    # variables = X.columns.values
    # imp = permutation_importance(rf, X, y, n_repeats=10, n_jobs=4)
    # # los índices ordenados de los valores mayores que 0.01, máximo 10
    # indices = [ind for ind in imp.importances_mean.argsort()[::-1] if imp.importances_mean[ind] >= 0.01][:10]
    # plt.bar(variables[indices], imp.importances_mean[indices], yerr=imp.importances_std[indices], capsize=5)
    # plt.xticks(rotation=35, ha='right', rotation_mode='anchor')
    # plt.show()
    #
    # # visualizar el primer árbol
    # nombre_arbol = 'tree'
    # export_graphviz(rf.estimators_[0], out_file=f'{nombre_arbol}.dot',
    #                 feature_names=X.columns,
    #                 class_names=['HC', 'IDIOPATHIC PD'],
    #                 rounded=True, proportion=False,
    #                 precision=2, filled=True)
    #
    # os.system(f'dot -Tpng {nombre_arbol}.dot -o {nombre_arbol}.png')
    # os.remove(f'{nombre_arbol}.dot')

    # gráfico de accuracy según el peso para HC
    px.append(p)
    py.append(acc_rf_bueno)

plt.plot(px, py), plt.show()
print('p:', px[py.index(max(py))])
print('acc:', max(py))

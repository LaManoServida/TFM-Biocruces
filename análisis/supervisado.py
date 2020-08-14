import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer
from sklearn.tree import export_graphviz
import pickle
from matplotlib import pyplot as plt

# PARÁMETROS
ruta_archivo = 'D:/Dropbox/UNI/TFM/datos/14 - Juntar HC e IDIOPATHIC PD/HC + IDIOPATHIC PD.csv'
lista_negra = ['PATNO', 'age_dis_onset', 'NP3SPCH', 'NP3FACXP', 'NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'PN3RIGRL',
               'NP3RIGLL', 'NP3FTAPR', 'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL', 'NP3TTAPR',
               'NP3TTAPL', 'NP3LGAGR', 'NP3LGAGL', 'NP3RISNG', 'NP3GAIT', 'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR',
               'NP3BRADY', 'NP3PTRMR', 'NP3PTRML', 'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL',
               'NP3RTALL', 'NP3RTALJ', 'NP3RTCON', 'DYSKPRES', 'NHY', 'NUPSOURC2Q', 'NP2SPCH', 'NP2SALV', 'NP2SWAL',
               'NP2EAT', 'NP2DRES', 'NP2HYGN', 'NP2HWRT', 'NP2HOBB', 'NP2TURN', 'NP2TRMR', 'NP2RISE', 'NP2WALK',
               'NP2FREZ', 'NUPSOURC1', 'NP1COG', 'NP1HALL', 'NP1DPRS', 'NP1ANXS', 'NP1APAT', 'NP1DDS', 'NUPSOURC1Q',
               'NP1SLPN', 'NP1SLPD', 'NP1PAIN', 'NP1URIN', 'NP1CNST', 'NP1LTHD', 'NP1FATG']
variables_numericas = ['HVLTRT1', 'HVLTRT2', 'HVLTRT3', 'HVLTRDLY', 'HVLTREC', 'MCAVFNUM', 'MCATOT', 'EDUCYRS',
                       'SDMTOTAL', 'UPSITBK1', 'UPSITBK2', 'UPSITBK3', 'UPSITBK4', 'age']
clase = 'Class'
semilla = 0
hacer_grid_search = True
beta = 1.2  # parámetro para el F-beta score

# leer datos
datos = pd.read_csv(ruta_archivo, sep=',', float_precision='round_trip')

# quitar variables de la lista negra
datos.drop(lista_negra, axis=1, inplace=True)

# quitar variables que no varían xd
datos.drop(datos.columns[datos.nunique().values == 1], axis=1, inplace=True)

# codificar la variable ENROLL_STATUS con etiquetas numéricas
datos['ENROLL_STATUS'] = LabelEncoder().fit_transform(datos['ENROLL_STATUS'])

# codificar en one hot las variables categóricas
# (que son todas excepto las numéricas, las que solo tengan dos valores únicos y la clase)
variables_no_onehot = variables_numericas + datos.columns[datos.nunique().values == 2].to_list() + [clase]
cols_categoricas = datos[datos.columns.difference(variables_no_onehot)]
one_hot = OneHotEncoder()
cat_onehot = pd.DataFrame(one_hot.fit_transform(cols_categoricas).toarray())
cat_onehot.columns = one_hot.get_feature_names(cols_categoricas.columns)  # cambiarles los nombres

# sustituir su versión anterior
datos.drop(cols_categoricas.columns, axis=1, inplace=True)
datos = datos.join(cat_onehot)

# separar atributos de clase
X = datos.drop(clase, axis=1)
y = datos[clase]

# separar train y test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=semilla, stratify=y.values)

# entrenar baseline: árbol de clasificación y sacar predicciones
bl_y_pred = cross_val_predict(DecisionTreeClassifier(random_state=semilla), X, y)  # 5-fold CV

# TODO: podar post entrenamiento
# TODO: no es lo mismo FP que FN
# TODO: qué es la importancia y cuánto debería ser o
# TODO: cómo calcula la de todos los árboles
# TODO: la importancia de variables tiene en cuenta los hiperparámetros?

if hacer_grid_search:
    # obtener el modelo random forest con mejores valores para los hiperparámetros con grid search
    valores_buscar_params = {'n_estimators': [100, 200],
                             'criterion': ['gini', 'entropy'],
                             'min_samples_split': [2, 3, 4, 5, 6, 7],
                             'min_samples_leaf': [1, 2, 3, 4, 5],
                             'max_depth': [5, 6, 7, 8, 9, 10],  # TODO poner menos de 5 un poquito
                             'max_features': ['sqrt'],
                             'min_impurity_decrease': [0.001]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=semilla), valores_buscar_params,
                               scoring=make_scorer(fbeta_score, beta=beta, pos_label='IDIOPATHIC PD'), n_jobs=4,
                               verbose=3)
    grid_search.fit(X, y)  # probar y entrenar todas las combinaciones con 5-fold CV
    rf = grid_search.best_estimator_
else:
    # entrenar random forest con los hiperparámetros especificados
    valores_params = {'n_estimators': 200,
                      'criterion': 'gini',
                      'min_samples_split': 2,
                      'min_samples_leaf': 2,
                      'max_depth': 5,
                      'max_features': 'sqrt',
                      'min_impurity_decrease': 0.001}
    rf = RandomForestClassifier(**valores_params, random_state=semilla)
    grid_search = None

# sacar predicciones con CV y accuracy
y_pred = cross_val_predict(rf, X, y)  # 5-fold CV
acc_rf_bueno = round(fbeta_score(y, y_pred, beta=beta), 4)

# volver a entrenarlo pero ahora con todos los datos
rf.fit(X, y)

# guardarlo
pickle.dump(rf, open(f'Modelo {acc_rf_bueno}.pickle', 'wb'))
# loaded_model = pickle.load(open('', 'rb'))

# evaluar
print('[BASELINE - ÁRBOL DE CLASIFICAICÓN]')
print('Matriz de confusión:\n', confusion_matrix(y, bl_y_pred))
print('Fbetaasdas-:', round(fbeta_score(y, bl_y_pred, beta=beta), 4), '\n')

print('[RANDOM FOREST]')
if hacer_grid_search:
    print(f'Grid search - Mejores parámetros encontrados: {grid_search.best_params_}')
print('Matriz de confusión:\n', confusion_matrix(y, y_pred))
print('fbetasasaa:', acc_rf_bueno, '\n')

# visualizar la importancia de las variables
variables = X.columns.values
importancias = rf.feature_importances_
plt.bar(variables[importancias.argsort()][:15], sorted(importancias, reverse=True)[:15])
plt.xticks(rotation=35, ha='right', rotation_mode='anchor')
plt.show()

# visualizar un árbol
nombre_arbol = 'tree'
export_graphviz(rf.estimators_[0], out_file=f'{nombre_arbol}.dot',  # TODO bueno por qué este árbol xd
                feature_names=X.columns,
                class_names=['HC', 'IDIOPATHIC PD'],
                rounded=True, proportion=False,
                precision=2, filled=True)

os.system(f'dot -Tpng {nombre_arbol}.dot -o {nombre_arbol}.png')
os.remove(f'{nombre_arbol}.dot')

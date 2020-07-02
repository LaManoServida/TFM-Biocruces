import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz
import os

# PARÁMETROS
ruta_archivo = 'D:/Dropbox/UNI/TFM/datos/14 - Juntar HC e IDIOPATHIC PD/HC + IDIOPATHIC PD.csv'
clase = 'Class'
variables_numericas = ['HVLTRT1', 'HVLTRT2', 'HVLTRT3', 'HVLTRDLY', 'HVLTREC', 'MCAVFNUM', 'MCATOT', 'EDUCYRS',
                       'SDMTOTAL', 'UPSITBK1', 'UPSITBK2', 'UPSITBK3', 'UPSITBK4', 'age'] + [clase]
semilla = 0

# leer datos
datos = pd.read_csv(ruta_archivo, sep=',', float_precision='round_trip')

# quitar variables PATNO y age_dis_onset
datos.drop(['PATNO', 'age_dis_onset'], axis=1, inplace=True)

# codificar la variable ENROLL_STATUS con etiquetas numéricas
datos['ENROLL_STATUS'] = LabelEncoder().fit_transform(datos['ENROLL_STATUS'])

# codificar las variables categóricas en one hot
cols_categoricas = datos[datos.columns.difference(variables_numericas)]
oh = OneHotEncoder()
cat_onehot = pd.DataFrame(oh.fit_transform(cols_categoricas).toarray())
cat_onehot.columns = oh.get_feature_names(cols_categoricas.columns)  # cambiarles los nombres

# sustituirlas por su versión anterior
datos.drop(cols_categoricas.columns, axis=1, inplace=True)
datos = datos.join(cat_onehot)

# separar atributos de clase y train de test
X = datos.drop(clase, axis=1)
y = datos[clase]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=semilla)

# entrenar random forest
rf = RandomForestClassifier(n_estimators=10, random_state=semilla)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# evaluar
print('Matriz de confusión:\n', confusion_matrix(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))

# visualizar un árbol
# TODO: nombre de la clase mal puesto
# TODO: feature importances rf.estimators_[0].feature_importances_
export_graphviz(rf.estimators_[0], out_file='tree.dot',
                feature_names=X.columns,
                class_names=clase,
                rounded=True, proportion=False,
                precision=2, filled=True)

os.system('dot -Tpng tree.dot -o tree.png')

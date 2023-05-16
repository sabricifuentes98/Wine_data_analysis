#Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Leer el conjunto de datos
wine_data = pd.read_csv('C:/Users/scifuenl/Desktop/WineQT.csv')

#Visualizar los primeros registros del conjunto de datos
print(wine_data.head())

#Comprobar la calidad de los datos
print(wine_data.isnull().sum())
wine_data.describe()

#Eliminar 'Id'
wine_data = wine_data.drop('Id', axis=1)

#Número de muestras para cada valor de calidad
counts = wine_data['quality'].value_counts()
print(counts)

#Analisis de correlacion
#Variables más influyentes en la calidad del vino
corr_matrix = wine_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.figure(figsize=(12, 8))
plt.show()

#Analisis de PCA
#Variables mas influyentes en la calidad del vino
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Estandarizamos los datos
scaler = StandardScaler()
wine_data_std = scaler.fit_transform(wine_data)

#Creamos el modelo PCA y obtenemos los componentes principales
pca = PCA()
pca.fit(wine_data_std)

#Mostramos la varianza explicada por componente principal
print('Varianza explicada por cada componente principal:', pca.explained_variance_ratio_)

#Transformar los datos al espacio de PCA
X_pca = pca.transform(X)

#Crear dataframe con las dos primeras componentes principales
pca_df = pd.DataFrame(data=X_pca[:,0:2], columns=['PC1', 'PC2'])
pca_df['class'] = y.values
print(pca_df.head())

#Visualizar resultados d PCA
#Extraer las dos primeras componentes principales
PC1 = pca_df['PC1']
PC2 = pca_df['PC2']

#Crear el gráfico de dispersión y pintar los puntos según la calidad del vino
plt.scatter(PC1, PC2, c=wine_data['quality'], cmap='coolwarm')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Análisis de Componentes Principales')
plt.colorbar()

plt.show()

#Analisis de importancia con Random Forest para determinar variables influyentes en la calidad del vino
from sklearn.ensemble import RandomForestRegressor

#Separar las variables predictoras y la variable objetivo
X = wine_data.drop(['quality'], axis=1)
y = wine_data['quality']

#Crear modelo 
rf = RandomForestRegressor(n_estimators=10, random_state=42)

#Entrenar modelo
rf.fit(X, y)

#Obtener la importancia de cada variable
importances = rf.feature_importances_
indices = importances.argsort()[::-1]

print("Importancia de las variables:")
for i in range(X.shape[1]):
    print("%d) %s (%f)" % (i + 1, X.columns[indices[i]], importances[indices[i]]))

#Prueba hipotesis t-student para determinar variables influyentes en la calidad del vino
from scipy import stats

#Crear grupos para cada valor de calidad
q3 = wine_data[wine_data['quality'] == 3].iloc[:, :-1]
q4 = wine_data[wine_data['quality'] == 4].iloc[:, :-1]
q5 = wine_data[wine_data['quality'] == 5].iloc[:, :-1]
q6 = wine_data[wine_data['quality'] == 6].iloc[:, :-1]
q7 = wine_data[wine_data['quality'] == 7].iloc[:, :-1]
q8 = wine_data[wine_data['quality'] == 8].iloc[:, :-1]

#Calcular la prueba t y el valor p para cada variable
p_values = []
for column in wine_data.columns[:-1]:
    t3, t4, t5, t6, t7, t8 = np.mean(q3[column]), np.mean(q4[column]), np.mean(q5[column]), np.mean(q6[column]), np.mean(q7[column]), np.mean(q8[column])
    n = len(q3[column]) + len(q4[column]) + len(q5[column]) + len(q6[column]) + len(q7[column]) + len(q8[column])
    s = np.sqrt(((len(q3[column])-1)*np.var(q3[column]) + (len(q4[column])-1)*np.var(q4[column]) + (len(q5[column])-1)*np.var(q5[column]) + (len(q6[column])-1)*np.var(q6[column]) + (len(q7[column])-1)*np.var(q7[column]) + (len(q8[column])-1)*np.var(q8[column]))/(n-6))
    t_stat = (t3 - t4) / (s * np.sqrt(1/len(q3[column]) + 1/len(q4[column])))
    p_value = 2*(1 - stats.t.cdf(abs(t_stat), df=n-6))
    p_values.append(p_value)

#Crear un dataframe con los resultados de la prueba
ttest_results = pd.DataFrame({'variable': wine_data.columns[:-1], 'p-value': p_values})
ttest_results = ttest_results.sort_values('p-value')

#Relacion entre la acidez volatil y la calidad
#Definir los intervalos para la variable 'volatile acidity'
bins = [0, 0.5, 1, 1.5, 2, 2.5, 3]

#Agrupar los datos por intervalos de 'volatile acidity' y calcular la media de 'quality' para cada intervalo
wine_data['va_interval'] = pd.cut(wine_data['volatile acidity'], bins)
grouped = wine_data.groupby('va_interval', as_index=False)['quality'].mean()

sns.barplot(x='va_interval', y='quality', data=grouped)
plt.xlabel('Rango de volatile acidity')
plt.ylabel('Calidad del vino (media)')
plt.title('Relación entre volatile acidity y calidad del vino')
plt.show()

#Relacion entre el acido sulfurico total y la calidad
bins = [0, 10, 20, 30, 40, 50, 60, 70]
wine_data['tsd_interval'] = pd.cut(wine_data['total sulfur dioxide'], bins)
grouped = wine_data.groupby('tsd_interval', as_index=False)['quality'].mean()

sns.barplot(x='tsd_interval', y='quality', data=grouped)
plt.xlabel('Rango de total sulfur dioxide')
plt.ylabel('Calidad del vino')
plt.title('Relación entre total sulfur dioxide y calidad del vino')
plt.show()

#Relacion entre el alcohol y la calidad
bins = [8, 9, 10, 11, 12, 13, 14, 15]
wine_data['alcohol_interval'] = pd.cut(wine_data['alcohol'], bins)
grouped = wine_data.groupby('alcohol_interval', as_index=False)['quality'].mean()

sns.barplot(x='alcohol_interval', y='quality', data=grouped)
plt.xlabel('Rango de alcohol')
plt.ylabel('Calidad del vino (media)')
plt.title('Relación entre alcohol y calidad del vino')
plt.show()

#Relacion entre los sulfatos y la calidad
bins = [0, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
wine_data['sulphates_interval'] = pd.cut(wine_data['sulphates'], bins)
grouped = wine_data.groupby('sulphates_interval', as_index=False)['quality'].mean()

sns.barplot(x='sulphates_interval', y='quality', data=grouped)
plt.xlabel('Rango de sulphates')
plt.ylabel('Calidad del vino (media)')
plt.title('Relación entre sulphates y calidad del vino')
plt.show()

#Relacion entre el azucar residual y la calidad
bins = [0, 2, 4, 6, 8, 10, 12]
wine_data['Rango de residual sugar'] = pd.cut(wine_data['residual sugar'], bins)
grouped = wine_data.groupby('Rango de residual sugar', as_index=False)['quality'].mean()

sns.barplot(x='Rango de residual sugar', y='quality', data=grouped)
plt.xlabel('Rango de residual sugar')
plt.ylabel('Calidad del vino')
plt.title('Relación entre residual sugar y calidad del vino')
plt.show()

#Equilibrar conjunto de datos con sobremuestreo
import warnings
warnings.filterwarnings('ignore')

counts = wine_data['quality'].value_counts()
#Calcular la cantidad de muestras adicionales necesarias para igualar la cantidad de vinos de calidad 5 y 4
n_samples = counts[5] - counts[4]

#Obtener todas las filas con calidad 4 y 3
data_4 = wine_data[wine_data['quality'] == 4]
data_3 = wine_data[wine_data['quality'] == 3]

#Muestrear aleatoriamente filas de calidad 4 y 3 para agregar al DataFrame original
for i in range(n_samples):
    row = pd.concat([data_4.sample(n=1), data_3.sample(n=1)])
    wine_data = wine_data.append(row)

# Verificar que las cantidades de cada calidad
print(wine_data['quality'].value_counts())

corr_matrix_equilibrado = wine_data.corr()
sns.heatmap(corr_matrix_equilibrado, annot=True, cmap='coolwarm', center=0)
plt.figure(figsize=(12, 8))
plt.show()

#Relacion entre el alcohol y la calidad del vino despues del sobremuestreo
bins = [8, 9, 10, 11, 12, 13, 14, 15]
wine_data['alcohol_interval'] = pd.cut(wine_data['alcohol'], bins)
grouped = wine_data.groupby('alcohol_interval', as_index=False)['quality'].mean()

sns.barplot(x='alcohol_interval', y='quality', data=grouped)
plt.xlabel('Rango de alcohol')
plt.ylabel('Calidad del vino (media)')
plt.title('Relación entre alcohol y calidad del vino')
plt.show()

#Relacion entre el azucar residual y la calidad del vino despues del sobremuestreo
bins = [0, 2, 4, 6, 8, 10, 12]
wine_data['Rango de residual sugar'] = pd.cut(wine_data['residual sugar'], bins)
grouped = wine_data.groupby('Rango de residual sugar', as_index=False)['quality'].mean()

sns.barplot(x='Rango de residual sugar', y='quality', data=grouped)
plt.xlabel('Rango de residual sugar')
plt.ylabel('Calidad del vino')
plt.title('Relación entre residual sugar y calidad del vino')
plt.show()

#Relacion entre la acidez volatil y la calidad del vino despues del sobremuestreo
#Definir los intervalos para la variable 'volatile acidity'
bins = [0, 0.5, 1, 1.5, 2, 2.5, 3]

#Agrupar los datos por intervalos de 'volatile acidity' y calcular la media de 'quality' para cada intervalo
wine_data['va_interval'] = pd.cut(wine_data['volatile acidity'], bins)
grouped = wine_data.groupby('va_interval', as_index=False)['quality'].mean()

sns.barplot(x='va_interval', y='quality', data=grouped)
plt.xlabel('Rango de volatile acidity')
plt.ylabel('Calidad del vino (media)')
plt.title('Relación entre volatile acidity y calidad del vino')
plt.show()

# Separar variables predictoras y variable objetivo
X = wine_data.drop(['quality'], axis=1)
y = wine_data['quality']

#Analisis de correlacion con datos equilibrados
corr = wine_data.corr()

plt.figure()
sns.heatmap(corr[['quality']].sort_values(by='quality', ascending=False),
            vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title("Correlación de las variables con la calidad")
plt.show()

#REGRESION LINEAL
from sklearn.linear_model import LinearRegression

#Variables de interés
X = wine_data[['fixed acidity','volatile acidity','citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide','density', 'pH','sulphates','alcohol']]
y = wine_data['quality']

#Crear modelo y ajustarlo
modelo = LinearRegression()
modelo.fit(X, y)

#Visualizacion del modelo de regresion lineal
import matplotlib.pyplot as plt

#Predicciones del modelo
y_pred_regresion = modelo.predict(X)

fig, ax = plt.subplots()
#Puntos reales
ax.scatter(y, y, c='blue', label='Real')
#Puntos predichos
ax.scatter(y, y_pred_regresion, c='red', label='Predicción')
#Línea para mostrar el ajuste del modelo
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)

ax.legend()
ax.set_xlabel('Calidad real')
ax.set_ylabel('Calidad predicha')

#Prediccion con Regresion lineal
nuevo_vino_regresion = [[5.6, 0.85, 0.05, 1.4, 0.045, 12, 88, 0.99,3.56, 0.77, 10.8]]
calidad_predicha_regresion = modelo.predict(nuevo_vino_regresion)
print(f"La calidad del vino predicha es: {calidad_predicha_regresion[0]}")

#Cambiado: acidez volatil y alcohol
nuevo_vino3_regresion = [[5.6, 0.05, 0.05, 1.4, 0.045, 12, 88, 0.99,3.56, 0.77, 6]]
nuevo_vino4_regresion = [[5.6, 0.35, 0.05, 1.4, 0.045, 12, 88, 0.99,3.56, 0.77, 12.6]]

calidad_predicha_regresion3 = modelo.predict(nuevo_vino3_regresion)
calidad_predicha_regresion4 = modelo.predict(nuevo_vino4_regresion)

print(f"La calidad del vino 3 predicha es: {calidad_predicha_regresion3[0]}")
print(f"La calidad del vino 4 predicha es: {calidad_predicha_regresion4[0]}")

#Cambiado: fixed acidity, residual sugar, total sulphure dioxide
nuevo_vino5_regresion = [[5.6, 0.35, 0.05, 1.4, 0.045, 12, 88, 0.99,3.56, 0.77, 12.6]]

calidad_predicha_regresion5 = modelo.predict(nuevo_vino5_regresion)

print(f"La calidad del vino 5 predicha es: {calidad_predicha_regresion5[0]}")

#Predicciones sobre la calidad del vino
wine_test = wine_data

X_test = wine_test[['volatile acidity', 'alcohol', 'sulphates', 'citric acid', 'free sulfur dioxide']]
y_pred = modelo.predict(X_test)

#Metricas de rendimiento Regresion lineal
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#Calcular R^2
r2 = r2_score(y, y_pred)
print('r2:', r2)

#Calcular MSE (Error cuadratico medio)
mse = mean_squared_error(y, y_pred)
print('mse: ', mse)

#Calcular RMSE (Raiz del error cuadratico medio)
rmse = mean_squared_error(y, y_pred, squared=False)
print('rmse: ', rmse)

#Calcular MAE (Error absoluto medio)
mae = mean_absolute_error(y, y_pred)
print('MAE:', mae)

#ARBOL DE DECISION
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

data_tree = wine_data
X_tree = data_tree.iloc[:, :-1]  # características
y_tree = data_tree.iloc[:, -1]  # variable objetivo
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train_tree, y_train_tree)

y_pred_tree = tree_model.predict(X_test_tree)

#Evaluar Arbol de Decision
r2_tree = r2_score(y_test_tree, y_pred_tree)
mse_tree = mean_squared_error(y_test_tree, y_pred_tree)
rmse_tree = mean_squared_error(y_test_tree, y_pred_tree, squared=False)
mae_tree = mean_absolute_error(y_test_tree, y_pred_tree)

print("R^2: ", r2_tree)
print("MSE: ", mse_tree)
print("RMSE: ", rmse_tree)
print("MAE: ", mae_tree)

#Visualizacion de arbol de decision
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(tree_model, out_file=None, 
                           feature_names=X_tree.columns, 
                           filled=True, rounded=True, 
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('tree_model')

#Hiperparametros
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'criterion': ['gini', 'entropy']}

dt = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(dt, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train_tree, y_train_tree)

print("Los mejores parámetros son:", grid_search.best_params_)
print("El mejor score es:", grid_search.best_score_)

#Prediccion con Arbol de decision
nuevo_vino = np.array([5.6, 0.85, 0.05, 1.4, 0.045, 12, 88, 0.99,3.56, 0.77, 10.8])

calidad_predicha = tree_model.predict(nuevo_vino.reshape(1, -1))

print(f"La calidad del vino es: {calidad_predicha[0]}")

#Cambiado: acidez volatil y alcohol
nuevo_vino3 = np.array([5.6, 0.05, 0.05, 1.4, 0.045, 12, 88, 0.99,3.56, 0.77, 6])
nuevo_vino4 = np.array([5.6, 0.35, 0.05, 1.4, 0.045, 12, 88, 0.99,3.56, 0.77, 12.6])
#Cambiado: fixed acidity, residual sugar, total sulphure dioxide
nuevo_vino5 = np.array([7.6, 0.35, 0.05, 3.4, 0.045, 12, 50, 0.99,3.56, 0.77, 12.6])

calidad_predicha3 = tree_model.predict(nuevo_vino3.reshape(1, -1))
calidad_predicha4 = tree_model.predict(nuevo_vino4.reshape(1, -1))
calidad_predicha5 = tree_model.predict(nuevo_vino5.reshape(1, -1))

print(f"La calidad del vino 3 es: {calidad_predicha3[0]}")
print(f"La calidad del vino 4 es: {calidad_predicha4[0]}")
print(f"La calidad del vino 5 es: {calidad_predicha5[0]}")

# -*- coding: utf-8 -*-
"""RedesNeuronales_wine_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UCh6ZREZ7h_TS8DrBlwZJ0fsfSfYINTf
"""

import pandas as pd
wine_data = pd.read_csv('/content/drive/MyDrive/WineQT.csv')

wine_data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#Eliminar 'Id'
wine_data = wine_data.drop('Id', axis=1)

#Equilibrar conjunto de datos con sobremuestreo
import warnings
warnings.filterwarnings('ignore')

counts = wine_data['quality'].value_counts()
#Calcular la cantidad de muestras adicionales necesarias para igualar la cantidad de vinos de calidad 5 y 4
n_samples = counts[5] - counts[4]

#Obtener todas las filas con calidad 4 y 3
data_4 = wine_data[wine_data['quality'] == 4]
data_3 = wine_data[wine_data['quality'] == 3]

#Muestrear aleatoriamente filas de calidad 4 y 3 para agregar al DataFrame original
for i in range(n_samples):
    row = pd.concat([data_4.sample(n=1), data_3.sample(n=1)])
    wine_data = wine_data.append(row)

# Verificar que las cantidades de cada calidad
print(wine_data['quality'].value_counts())

X_redes = wine_data.iloc[:, :-1].values
y_redes = wine_data.iloc[:, -1].values

X_train_redes, X_test_redes, y_train_redes, y_test_redes = train_test_split(X_redes, y_redes, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=X_train_redes.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train_redes, y_train_redes, epochs=100, batch_size=32, verbose=0)

#Evaluacion del modelo en los datos de prueba
y_pred_redes = model.predict(X_test_redes)
r2 = r2_score(y_test_redes, y_pred_redes)
mse = mean_squared_error(y_test_redes, y_pred_redes)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_redes, y_pred_redes)

print("R^2: {:.2f}".format(r2))
print("MSE: {:.2f}".format(mse))
print("RMSE: {:.2f}".format(rmse))
print("MAE: {:.2f}".format(mae))

nuevo_vino_pred_redes = np.array([5.6, 0.85, 0.05, 1.4, 0.045, 12, 88, 0.99, 3.56, 0.77, 10.8])
calidad_predicha = model.predict(nuevo_vino_pred_redes.reshape(1, -1))
print(f"La calidad del vino predicha es: {calidad_predicha[0]}")

nuevo_vino_pred_redes3 = np.array([5.6, 0.05, 0.05, 1.4, 0.045, 12, 88, 0.99, 3.56, 0.77, 6])
nuevo_vino_pred_redes4 = np.array([5.6, 0.35, 0.05, 1.4, 0.045, 12, 88, 0.99, 3.56, 0.77, 12.6])

calidad_predicha3 = model.predict(nuevo_vino_pred_redes3.reshape(1, -1))
calidad_predicha4 = model.predict(nuevo_vino_pred_redes4.reshape(1, -1))

print(f"La calidad del vino 3 predicha es: {calidad_predicha3[0]}")
print(f"La calidad del vino 4 predicha es: {calidad_predicha4[0]}")

nuevo_vino_pred_redes5 = np.array([7.6, 0.35, 0.05, 3.4, 0.045, 12, 50, 0.99, 3.56, 0.77, 12.6])

calidad_predicha5 = model.predict(nuevo_vino_pred_redes5.reshape(1, -1))

print(f"La calidad del vino 5 predicha es: {calidad_predicha5[0]}")

#Funcion de activacion Sigmoide

# Dividir los datos en características (X) y etiquetas (y)
X_sigmoide = wine_data.drop('quality', axis=1)
y_sigmoide = wine_data['quality']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_sigmoide, X_test_sigmoide, y_train_sigmoide, y_test_sigmoide = train_test_split(X_sigmoide, y_sigmoide, test_size=0.2, random_state=42)

# Escalar los datos de entrenamiento
scaler = StandardScaler()
X_train_sigmoide = scaler.fit_transform(X_train_sigmoide)

# Definir el modelo
model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(X_train_sigmoide.shape[1],)))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train_sigmoide, y_train_sigmoide, epochs=100, batch_size=32)

# Escalar los datos de prueba
X_test_sigmoide = scaler.transform(X_test_sigmoide)

# Evaluar el modelo con los datos de prueba
y_pred_sigmoide = model.predict(X_test_sigmoide)
r2_sigmoide = r2_score(y_test_sigmoide, y_pred_sigmoide)
mse_sigmoide = mean_squared_error(y_test_sigmoide, y_pred_sigmoide)
rmse_sigmoide = np.sqrt(mse_sigmoide)
mae_sigmoide = mean_absolute_error(y_test_sigmoide, y_pred_sigmoide)

print("R^2: {:.2f}".format(r2_sigmoide))
print("MSE: {:.2f}".format(mse_sigmoide))
print("RMSE: {:.2f}".format(rmse_sigmoide))
print("MAE: {:.2f}".format(mae_sigmoide))






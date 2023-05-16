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

#Verificar que las cantidades de cada calidad
print(wine_data['quality'].value_counts())

#Separar variables predictoras y variable objetivo
X = wine_data.drop(['quality'], axis=1)
y = wine_data['quality']

#Modelo de aprendizaje automatico REGRESION LINEAL
from sklearn.linear_model import LinearRegression

#Variables de interés
X = wine_data[['fixed acidity','volatile acidity','citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide','density', 'pH','sulphates','alcohol']]
y = wine_data['quality']

#Crear modelo de regresion lineal y ajustarlo
modelo = LinearRegression()
modelo.fit(X, y)

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

#Arbol de decision
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

#Metricas de rendimiento Arbol de Decision

r2_tree = r2_score(y_test_tree, y_pred_tree)
mse_tree = mean_squared_error(y_test_tree, y_pred_tree)
rmse_tree = mean_squared_error(y_test_tree, y_pred_tree, squared=False)
mae_tree = mean_absolute_error(y_test_tree, y_pred_tree)

print("R^2: ", r2_tree)
print("MSE: ", mse_tree)
print("RMSE: ", rmse_tree)
print("MAE: ", mae_tree)

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
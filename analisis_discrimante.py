from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Cargar el dataset iris
iris = datasets.load_iris()

# Convertir el dataset a un DataFrame de pandas
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']

# Ver las primeras seis filas del DataFrame
print(df.head())

#find how many total observations are in dataset
print(len(df.index))


#define predictor and response variables
X = df[['s_length', 's_width', 'p_length', 'p_width']]
y = df['species']

print(X)
#Fit the LDA model
model = LinearDiscriminantAnalysis()
model.fit(X, y)


#Define el método para evaluar el modelo
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#evaluar el modelo
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(np.mean(scores))   


#define nueva observación
new = [5, 3, 1, .4]

#predice a qué clase pertenece la nueva observación
print(model.predict([new]))





#definir datos para graficar
X = iris.data
y = iris.target
model = LinearDiscriminantAnalysis()
data_plot = model.fit(X, y).transform(X)
target_names = iris.target_names

# gráfico LDA
plt.figure()
colors = ['red', 'green', 'blue']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_plot[y == i, 0], data_plot[y == i, 1], alpha=.8, color=color,
                label=target_name)

# agregar leyenda al gráfico
plt.legend(loc='best', shadow=False, scatterpoints=1)

# gráfico LDA mostrar
plt.show()







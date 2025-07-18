import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib.colors import ListedColormap

# Cargar el dataset Iris (usamos solo las dos primeras características)
iris = load_iris()
X = iris.data[:, :2]  # Solo tomamos las dos primeras columnas para poder graficar
y = iris.target

# Ajustar el modelo QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)

# Crear un grid de puntos en 2D para graficar las fronteras de decisión
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predecir en el grid para obtener las fronteras
Z = qda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Colores para las clases y fondo
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['red', 'green', 'blue']

# Graficar el fondo con las fronteras de decisión
plt.figure()
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Graficar los puntos de los datos originales
for i, color in zip(range(len(iris.target_names)), cmap_bold):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                edgecolor='black', s=20)

# Etiquetas y título
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Fronteras de decisión con QDA (Iris dataset)')
plt.legend(loc='best')

# Mostrar el gráfico
plt.show()

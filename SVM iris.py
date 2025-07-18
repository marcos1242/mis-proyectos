import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos de Iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # Usamos las dos primeras características para facilitar la visualización
y = iris.target

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Función para graficar límites de decisión
def plot_svm_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.show()

# Entrenar y graficar SVM con margen duro
svm_hard = SVC(kernel='linear', C=1).fit(X_train, y_train) # Entrena un SVM con kernel lineal y un valor de C muy alto. Esto genera un margen duro (hard margin), donde el modelo trata de separar perfectamente las clases, no permitiendo errores de clasificación
plot_svm_decision_boundary(svm_hard, X, y)     

# Entrenar y graficar SVM con margen suave
svm_soft = SVC(kernel='linear', C=0.01).fit(X_train, y_train)
plot_svm_decision_boundary(svm_soft, X, y)

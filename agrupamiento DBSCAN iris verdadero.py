import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Cargar el dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names  # Nombres de las clases

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducir la dimensionalidad con PCA a 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_pca)

# Visualización
plt.figure(figsize=(12, 6))

# Graficar los datos originales (después de PCA)
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
handles = [plt.Line2D([0], [0], marker='o', color=plt.cm.viridis(i / 2), linestyle='None', markersize=10)
           for i in range(3)]
labels_original = [f'Clase {name}' for name in target_names]
plt.title('Datos originales después de PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend(handles, labels_original, title='Clases Iris', loc='best')

# Graficar los clusters detectados por DBSCAN
plt.subplot(1, 2, 2)
unique_labels = set(labels)
colors = [plt.cm.viridis(each) for each in np.linspace(0, 1, len(unique_labels))]
for label, color in zip(unique_labels, colors):
    if label == -1:  # Ruido
        cluster_label = "Ruido"
        marker = 'x'
    else:
        cluster_label = f"Cluster {label}"
        marker = 'o'
    plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], color=color,
                marker=marker, label=cluster_label)
plt.title('Clusters detectados por DBSCAN')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend(title="Clusters detectados", loc='best')

# Mostrar gráficos
plt.tight_layout()
plt.show()

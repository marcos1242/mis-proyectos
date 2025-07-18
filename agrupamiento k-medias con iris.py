from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Cargar el dataset Iris
iris = load_iris()
X = iris.data  # Características de las flores (longitud y ancho de sépalos y pétalos)
y = iris.target  # Etiquetas reales (solo para comparación, no las usaremos en el modelo)

# Aplicar K-means con 3 clústeres (porque sabemos que hay 3 especies)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Resultados del modelo
labels = kmeans.labels_  # Etiquetas asignadas por K-means
print(labels)
centroids = kmeans.cluster_centers_  # Coordenadas de los centros de los clústeres
print(centroids)
# Crear un DataFrame para analizar los resultados
df = pd.DataFrame(X, columns=iris.feature_names)
print(df)
df['Cluster'] = labels  # Etiquetas asignadas por K-means
df['Actual'] = y  # Etiquetas reales (solo para comparación)
print(df)

print("Centros de los clústeres:")
print(centroids)

# Visualización: Usamos las primeras dos características (longitud y ancho de sépalo)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, label="Clústeres")
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.7, label='Centros')
plt.title('K-means en el Dataset Iris')
plt.xlabel('Longitud del sépalo (cm)')
plt.ylabel('Ancho del sépalo (cm)')
plt.legend()
plt.show()

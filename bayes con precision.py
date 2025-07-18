import pandas as pd  # Importar pandas para el manejo de datos
import numpy as np  # manejo de arreglos
import matplotlib.pyplot as plt  # gráficos
from sklearn.preprocessing import StandardScaler  # Escalado de datos
from sklearn.naive_bayes import GaussianNB  # clasificador bayesiano ingenuo
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score  # Añadimos accuracy_score

# Ruta con el nombre exacto del archivo
ruta_del_csv = r'C:\Users\marco\Downloads\inferencia estadistica\diabetes.csv'

# Lee el archivo CSV
df = pd.read_csv(ruta_del_csv)

# Muestra las primeras filas para verificar
print(df.head())

# Tamaño de gráficos
plt.rcParams["figure.figsize"] = (8, 8)

# Cargar los datos
datos = np.genfromtxt(ruta_del_csv, delimiter=',', skip_header=1)

# Elimina todas las filas con NaN
datos = datos[~np.isnan(datos).any(axis=1)]

# Extraer variables predictoras (features)
X = datos[:, 0:-1]

# Extraer clasificación (target)
y = datos[:, -1]

# Escalar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Crear y entrenar el modelo Naive Bayes
gnb = GaussianNB()
modelo_gnb = gnb.fit(X, y)  # ENTRENAMIENTO del modelo

# Predecir probabilidades
probas = gnb.predict_proba(X)

# Inspeccionar las probabilidades
with np.printoptions(precision=3, suppress=True):
    print(probas[0:20])

# Crear predicciones personalizadas basadas en un umbral
y_pred_07 = np.ones(y.shape)  # Asumo por defecto que todas las muestras pertenecen a la clase 1 (potable)
for i in range(probas.shape[0]):
    if (probas[i, 0] > 0.7):  # CAMBIAR
        y_pred_07[i] = 0      # CAMBIAR DEPENDIENDO DEL TIPO DE CLASIFICACION

# Matriz de confusión para las predicciones personalizadas
conf_07 = confusion_matrix(y, y_pred_07)

# Mostrar la matriz de confusión
disp_07 = ConfusionMatrixDisplay(confusion_matrix=conf_07, display_labels=gnb.classes_)
disp_07.plot(values_format='d')
plt.show()

# Calcular la precisión del modelo
accuracy = accuracy_score(y, y_pred_07)
print(f"\nExactitud del modelo Naive Bayes con umbral 0.7: {accuracy * 100:.2f}%")

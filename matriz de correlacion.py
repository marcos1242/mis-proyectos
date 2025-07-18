# Importar librerías necesarias
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar archivo desde Google Colab


# Subir el archivo desde tu computadora


# Leer el archivo CSV (asegúrate de que el nombre coincida con el archivo subido)
df = pd.read_csv(r'C:\Users\marco\Downloads\inferencia estadistica\Claseficación_banco.csv')

# Muestra las primeras filas para verificar
print(df.head())

# Comprobar si hay valores nulos y manejarlos (eliminar o imputar)
print("Valores nulos por columna:\n", df.isnull().sum())
df = df.dropna()  # Se eliminan filas con valores nulos (ajusta según sea necesario)

# Cantidad de observaciones en el dataset después de la limpieza
print(f"Número de observaciones después de la limpieza: {len(df.index)}")

# Extraer variables predictoras (features) - Todas las columnas menos las 2 primeras y la última
X = df.iloc[:, 2:-1]

# Extraer clasificación (target) - La última columna
y = df.iloc[:, -1]

# Normalizar datos
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Datos normalizados:\n", X.head())

# Ajustar el modelo
model = QuadraticDiscriminantAnalysis()
model.fit(X, y)

# Define el método para evaluar el modelo
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Evaluar el modelo
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(f"Precisión media del modelo: {np.mean(scores):.2f}")

# Definir la nueva observación en un DataFrame con las mismas columnas
new = pd.DataFrame([[2, 3, 4, 6, 4, 5, 6, 7, 8, 9, 5]],
                   columns=['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                            'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths',
                            'delinq.2yrs', 'pub.rec'])
print("Nueva observación:\n", new)

# Normalizar la nueva observación usando el mismo scaler
new_scaled = pd.DataFrame(scaler.transform(new), columns=new.columns)
print("Nueva observación normalizada:\n", new_scaled)

# Predice a qué clase pertenece la nueva observación
print("Predicción para la nueva observación:", model.predict(new_scaled))

# Crear gráfico de dispersión entre dos variables
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='int.rate', y='fico', hue='not.fully.paid', palette='coolwarm')
plt.title("Gráfico de Dispersión: int.rate - fico")
plt.xlabel('int.rate')
plt.ylabel('fico')
plt.show()

# Crear y visualizar la matriz de correlación solo con columnas numéricas
plt.figure(figsize=(12, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()  # Filtra solo columnas numéricas
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación")
plt.show()
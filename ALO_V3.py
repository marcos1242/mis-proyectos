import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import os
import sys
import argparse
import matplotlib.pyplot as plt  # Para graficar
import numpy as np

class CallPredictionModel:
    def __init__(self, csv_path, model_name="model.pkl", n_estimators=1000, learning_rate=0.1, max_depth=3):
        self.csv_path = csv_path
        self.model_path = os.path.join(os.path.dirname(__file__), model_name)
        self.model = None
        self.trained = False
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.load_model()
        self.data = None

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            self.trained = True
            print(f"Modelo cargado exitosamente desde {self.model_path}.")
        except FileNotFoundError:
            self.model = GradientBoostingRegressor(
                n_estimators=1,  # Empezamos con 1 estimador
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=1,
                warm_start=True  # Para entrenar en etapas
            )
            print(f"No se encontró un modelo guardado en {self.model_path}. Se inicializó uno nuevo.")

    def train(self):
        if self.trained:
            print("El modelo ya está entrenado. No se volverá a entrenar.")
            return
        try:
            df = pd.read_csv(self.csv_path).dropna()
            self.data = df
            X = df[['Fecha', 'Área', 'Hora']]
            y = df['Resultados']

            # Dividir los datos en entrenamiento y validación
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Listas para almacenar los valores de la función de pérdida
            train_loss = []
            val_loss = []

            # Entrenar el modelo en etapas
            for n_estimators in range(1, self.n_estimators + 1):
                self.model.n_estimators = n_estimators
                self.model.fit(X_train, y_train)

                # Calcular la pérdida en el conjunto de entrenamiento y validación
                train_loss.append(self.model.train_score_[-1])  # MSE en el conjunto de entrenamiento
                val_pred = self.model.predict(X_val)
                val_loss.append(np.mean((val_pred - y_val) ** 2))  # MSE en el conjunto de validación

            # Graficar la función de pérdida
            self.plot_loss_curve(train_loss, val_loss)

            # Guardar el modelo entrenado
            self.trained = True
            joblib.dump(self.model, self.model_path)
            print(f"Modelo entrenado y guardado en {self.model_path}.")
        except Exception as e:
            print(f"Error durante el entrenamiento del modelo: {e}")
            exit()

    def plot_loss_curve(self, train_loss, val_loss):
        """ Grafica la función de pérdida y la guarda en un archivo. """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.n_estimators + 1), train_loss, label='Pérdida en entrenamiento', color='blue')
        plt.plot(range(1, self.n_estimators + 1), val_loss, label='Pérdida en validación', color='red')
        plt.xlabel('Número de estimadores (árboles)')
        plt.ylabel('Error Cuadrático Medio (MSE)')
        plt.title('Función de Pérdida durante el Entrenamiento')
        plt.legend()
        plt.grid(True)

        # Guardar la gráfica en un archivo
        loss_curve_path = os.path.join(os.path.dirname(__file__), "loss_curve.png")
        plt.savefig(loss_curve_path)
        print(f"Gráfica de la función de pérdida guardada en {loss_curve_path}.")

        # Mostrar la gráfica
        plt.show()

    def predict(self, area, date, hour):
        features = pd.DataFrame([[date, area, hour]], columns=['Fecha', 'Área', 'Hora'])
        try:
            prediction = self.model.predict(features)[0]
            print(f"Predicción para la nueva observación {features.iloc[0].values}: {prediction}")
            return prediction
        except Exception as e:
            print(f"Ocurrió un error al predecir: {e}")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para realizar predicciones de llamadas.")
    parser.add_argument("-c", metavar="CODIGO_AREA", type=int, help="Código de área para consulta en el día y hora actuales.", nargs="?")
    parser.add_argument("-f", metavar=("CODIGO_AREA", "FECHA", "HORA"), type=str, help="Código de área, fecha (MMDD) y hora (HHMM) para la consulta.", nargs=3)
    
    args = parser.parse_args()
    
    # Ruta del archivo CSV
    ruta_del_csv = r"data.csv"
    
    # Crear una instancia del modelo
    model_instance = CallPredictionModel(csv_path=ruta_del_csv)
    
    # Entrenar el modelo si es necesario
    model_instance.train()
    
    if args.c:
        area = args.c
        current_datetime = datetime.now()
        date = current_datetime.strftime("%m%d")
        hour = current_datetime.strftime("%H%M")
        model_instance.predict(area, date, hour)
    
    elif args.f:
        try:
            area = int(args.f[0])
            date_str = args.f[1]
            hour_str = args.f[2]
            
            # Validar formato de fecha y hora
            if len(date_str) != 4 or len(hour_str) != 4:
                raise ValueError("Formato de fecha u hora incorrecto. Fecha debe ser MMDD y hora HHMM.")
            
            model_instance.predict(area, date_str, hour_str)
        except ValueError as e:
            print(f"Error: {e}")
    else:
        print("Error: Debe proporcionar el flag -c o -f con los argumentos correspondientes.")
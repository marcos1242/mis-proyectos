RESUMEN DE ALGUNOS DE LOS PROYECTOS


ALO_V3.py:

Desarrollé un sistema de predicción basado en Gradient Boosting Regressor para estimar la probabilidad de que una llamada sea aceptada en un día, hora y área determinados. El modelo se entrena automáticamente desde un CSV si no existe previamente, y grafica la curva de pérdida para evaluar su rendimiento. 
Incluye una interfaz por consola para consultar predicciones en tiempo real o a futuro



SVM iris.py:

Entrené y visualicé clasificadores SVM con kernel lineal sobre el dataset Iris. Comparé dos configuraciones distintas: una con margen duro (C alto) que busca separar las clases perfectamente, y otra con margen suave (C bajo) que permite errores para mejorar la generalización. 
Utilicé solo dos características para poder graficar las fronteras de decisión y mostrar cómo influye el parámetro C en el modelo.



agrupamiento DBSCAN iris verdadero.py:

Implementé un análisis de clustering no supervisado utilizando el dataset Iris. Apliqué escalado estándar, reducción de dimensionalidad con PCA, y agrupamiento con DBSCAN para identificar patrones sin etiquetas.
Comparé visualmente los clústeres detectados con las clases reales, destacando la capacidad del modelo para encontrar estructuras ocultas en los datos.

analisis_discrimante.py:

Implementé un clasificador de Análisis Discriminante Lineal (LDA) sobre el dataset Iris, evaluando su rendimiento con validación cruzada estratificada (RepeatedKFold). Realicé una predicción sobre una nueva observación y visualicé la proyección de los datos en el espacio LDA, mostrando cómo se separan las clases linealmente.

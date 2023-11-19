import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Cargar conjuntos de entrenamiento y prueba desde archivos CSV sin encabezados
train_data = pd.read_csv('XOR_trn.csv', header=None)  # Ajusta la ruta del archivo
test_data = pd.read_csv('XOR_tst.csv', header=None)  # Ajusta la ruta del archivo

# Asegurarse de que las columnas coincidan en ambos conjuntos
num_columns = min(train_data.shape[1], test_data.shape[1])

X_train = train_data.iloc[:, :num_columns - 1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :num_columns - 1]
y_test = test_data.iloc[:, -1]

# Inicializar y entrenar el modelo con más capas y neuronas
model = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
predictions_test = model.predict(X_test)

# Calcular la precisión del modelo en el conjunto de prueba
accuracy_test = accuracy_score(y_test, predictions_test)

print('Precisión en el conjunto de prueba: {:.2%}'.format(accuracy_test))

# Visualizar la frontera de decisión
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap=plt.cm.Spectral)
plt.title('Frontera de Decisión')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')

h = .02  # Tamaño de paso en la malla
x_min, x_max = X_test.iloc[:, 0].min() - 1, X_test.iloc[:, 0].max() + 1
y_min, y_max = X_test.iloc[:, 1].min() - 1, X_test.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.show()
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Inicialización del conjunto de datos con make_gaussian_quantiles
N = 1000  # muestras
gaussian_quantiles = make_gaussian_quantiles(mean=None,
    cov=0.1,
    n_samples=N,
    n_features=2,
    n_classes=2,
    shuffle=True,
    random_state=None
)

X, Y = gaussian_quantiles
Y = Y[:, np.newaxis]

# Split de datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Inicializar el modelo de regresión logística
model = LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, Y_train.ravel())

# Realizar predicciones en el conjunto de prueba
predictions = model.predict(X_test)

# Calcular la precisión del modelo en el conjunto de prueba
accuracy = accuracy_score(Y_test, predictions)
print(f'Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%')

# Crear n partículas aleatorias para la generalización
n_particulas = 100
particulas = np.random.rand(n_particulas, 2)

# Hacer predicciones con las partículas aleatorias
predicciones_particulas = model.predict(particulas)

# Imprimir las predicciones de las partículas aleatorias
print('Predicciones de las partículas aleatorias:')
print(predicciones_particulas)

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train.ravel(), cmap=plt.cm.Paired, edgecolors='k', marker='o', s=50)
plt.title('Datos de Entrenamiento y Frontera de Decisión')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')

# Plotear las partículas aleatorias y sus predicciones
plt.scatter(particulas[:, 0], particulas[:, 1], c=predicciones_particulas, cmap=plt.cm.Paired, edgecolors='k', marker='x', s=50)

# Mostrar la gráfica
plt.show()
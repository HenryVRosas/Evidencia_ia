## Trabajo

Adecuar el documento, y procesar el dataset de titanic


```python
# Importar librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
```


```python
# cargar el dataset como un dataframe
data = pd.read_csv("titanic.csv")  # Cambio de "student_data.csv" a "titanic.csv"
data[:5]  # Muestra los primeros 5 datos

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotear los datos
def plot_points(data):
    """Función para plotear los puntos de datos de supervivencia basado en 'Age' y 'Fare'."""
    X = np.array(data[["Age", "Fare"]])
    y = np.array(data["Survived"])
    survived = X[y == 1]
    not_survived = X[y == 0]
    plt.scatter(survived[:, 0], survived[:, 1], color='cyan', edgecolor='k', label='Survived')
    plt.scatter(not_survived[:, 0], not_survived[:, 1], color='red', edgecolor='k', label='Not Survived')
    plt.xlabel("Age (normalized)")
    plt.ylabel("Fare (normalized)")
    
plot_points(data)
plt.show()

```


    
![png](7_back_porpagation_titanic_files/7_back_porpagation_titanic_3_0.png)
    



```python
# Separar por Pclass
data_rank1 = data[data["Pclass"] == 1]  
data_rank2 = data[data["Pclass"] == 2]
data_rank3 = data[data["Pclass"] == 3]
data_rank4 = data[data["Pclass"] == 4]

# Plotear los grupos por clase
plot_points(data_rank1)
plt.title("Pclass 1")
plt.show()
plot_points(data_rank2)
plt.title("Pclass 2")
plt.show()
plot_points(data_rank3)
plt.title("Pclass 3")
plt.show()
plot_points(data_rank4)
plt.title("Pclass 4")
plt.show()

```


    
![png](7_back_porpagation_titanic_files/7_back_porpagation_titanic_4_0.png)
    



    
![png](7_back_porpagation_titanic_files/7_back_porpagation_titanic_4_1.png)
    



    
![png](7_back_porpagation_titanic_files/7_back_porpagation_titanic_4_2.png)
    



    
![png](7_back_porpagation_titanic_files/7_back_porpagation_titanic_4_3.png)
    



```python
# Aplicar one hot encoding y unir con columnas relevantes, se borro columna Pclass
data_encoded = pd.concat([data[['Survived', 'Age', 'Fare']], pd.get_dummies(data['Pclass'], prefix='Pclass')], axis=1)
for column in data_encoded.columns:
    data_encoded[column] = pd.to_numeric(data_encoded[column], errors='coerce').astype(np.float64)
data_encoded.dropna(inplace=True)
data_encoded
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>22.000000</td>
      <td>7.2500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>38.000000</td>
      <td>71.2833</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>26.000000</td>
      <td>7.9250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>35.000000</td>
      <td>53.1000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>35.000000</td>
      <td>8.0500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0.0</td>
      <td>27.000000</td>
      <td>13.0000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1.0</td>
      <td>19.000000</td>
      <td>30.0000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0.0</td>
      <td>29.699118</td>
      <td>23.4500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1.0</td>
      <td>26.000000</td>
      <td>30.0000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0.0</td>
      <td>32.000000</td>
      <td>7.7500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 6 columns</p>
</div>




```python
# Normalización y Escalamiento de 'Age' y 'Fare'
data_encoded['Age'] = data_encoded['Age'] / data_encoded['Age'].max()
data_encoded['Fare'] = data_encoded['Fare'] / data_encoded['Fare'].max()
data_encoded
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.275000</td>
      <td>0.014151</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.475000</td>
      <td>0.139136</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.325000</td>
      <td>0.015469</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.437500</td>
      <td>0.103644</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.437500</td>
      <td>0.015713</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0.0</td>
      <td>0.337500</td>
      <td>0.025374</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1.0</td>
      <td>0.237500</td>
      <td>0.058556</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0.0</td>
      <td>0.371239</td>
      <td>0.045771</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1.0</td>
      <td>0.325000</td>
      <td>0.058556</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0.0</td>
      <td>0.400000</td>
      <td>0.015127</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 6 columns</p>
</div>




```python
# Dividir los datos en conjuntos de entrenamiento y prueba es mejor que la 90 y 10
#Dividir la data entre x e y con train test split es mejor
X_train, X_test, y_train, y_test = train_test_split(data_encoded.drop('Survived', axis=1), data_encoded['Survived'], test_size=0.2, random_state=42)

```


```python
# Red neuronal
# Funciones
def sigmoid(x):
    """Aplica la función sigmoidal para activación."""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """Derivada de la función sigmoidal."""
    return sigmoid(x) * (1 - sigmoid(x))
```


```python
# Defino la red neuronal

epochs = 600
learnrate = 0.5

# Entrenamiento de la red, esta agregado backpropagation

def train_nn(features, targets, epochs, learnrate):
    """Entrena la red neuronal y muestra la pérdida en cada época."""
    np.random.seed(42)
    n_records, n_features = features.shape
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            output = sigmoid(np.dot(x, weights))
            error = -y * np.log(output) - (1 - y) * np.log(1 - output)
            error_term = (y - output) * sigmoid_prime(x)
            del_w += error_term * x
        weights += learnrate * del_w / n_records
        if e % (epochs / 10) == 0 or e == epochs - 1:
            predictions = sigmoid(np.dot(features, weights))
            loss = np.mean((predictions - targets) ** 2)
            print(f"Epoch {e + 1}, Loss: {loss}")
    
    print("=========")
    print("Finished training!")
    return weights

```


```python
# Entrenar el modelo y mostrar los pesos obtenidos
weights = train_nn(X_train, y_train, epochs, learnrate)
```

    Epoch 1, Loss: 0.25345458436520774
    Epoch 61, Loss: 0.21760579870145136
    Epoch 121, Loss: 0.21080200411124195
    Epoch 181, Loss: 0.2087781618592495
    Epoch 241, Loss: 0.20789352706472294
    Epoch 301, Loss: 0.20738343880908308
    Epoch 361, Loss: 0.2070294288048481
    Epoch 421, Loss: 0.20675181259775846
    Epoch 481, Loss: 0.20651611089468666
    Epoch 541, Loss: 0.20630577359775498
    Epoch 600, Loss: 0.2061154749653572
    =========
    Finished training!
    


```python
# Imprimo el ultimo peso obtenido
weights
```




    array([-0.7635315 ,  0.22320192,  0.69190107,  0.18817885, -0.92212221])




```python
# Evaluar la precisión del modelo

def calculate_accuracy(features, targets, weights):
    outputs = sigmoid(np.dot(features, weights))
    predictions = outputs > 0.5
    return np.mean(predictions == targets)

# Calcular y mostrar la precisión
training_accuracy = calculate_accuracy(X_train, y_train, weights)
testing_accuracy = calculate_accuracy(X_test, y_test, weights)
#print("Training Accuracy: {:.3f}".format(training_accuracy))
#print("Testing Accuracy: {:.3f}".format(testing_accuracy))

# Si deseas utilizar la variable 'accuracy' específicamente para la precisión de prueba y luego imprimir
accuracy = testing_accuracy
print("Prediction accuracy: {:.3f}".format(accuracy))

```

    Prediction accuracy: 0.721
    

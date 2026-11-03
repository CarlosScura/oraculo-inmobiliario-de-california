# El Oráculo Inmobiliario de California: La Venganza del RMSE

Segundo challenge de la especialización de datos de CodePRO.

En este proyecto se entrenan 2 modelos predictivos básicos de Machine Learning. El primer modelo es de regresión lineal, utiliza el dataset de [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) para hacer predicciones del precio mediano de las zonas de California.

El segundo modelo es de regresión logística, utiliza el dataset de [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) para hacer predicciones de qué clientes de Telco se dan de baja de sus servicios.

El desarrollo completo está en `oraculo_inmobiliario.ipynb`.


## Metodología

### Regresión lineal: precio de las zonas

1. **Datos:** `housing.csv`, con 20.640 zonas y 10 columnas.
2. **Preparación:** `ocean_proximity` es texto, así que cada una de sus 5 categorías se convirtió en una columna booleana (*one-hot encoding*). Se eliminó `INLAND` para que sea la categoría de referencia y los coeficientes tengan un valor único. `total_bedrooms` tiene 207 zonas con valores nulos, que se conservaron para no perder datos.
3. **División:** 80% para entrenamiento (16.512 zonas) y 20% para test (4.128 zonas), con `random_state=36`.
4. **Nulos:** se rellenaron después de dividir, con la mediana de `total_bedrooms` calculada solo con el entrenamiento (434), para que el test no influya en el entrenamiento (*data leakage*).
5. **Modelo y evaluación:** `LinearRegression`, evaluado con el RMSE sobre el test.
6. **Interpretación:** se leyeron los coeficientes de cada variable, manteniendo las demás iguales.


### Regresión logística: bajas de clientes

1. **Datos:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`, con 7.043 clientes y 21 columnas.
2. **Preparación:** se conservaron 6 columnas (`tenure`, `MonthlyCharges`, `SeniorCitizen`, `Contract`, `InternetService` y `PaperlessBilling`) y se descartaron `customerID` (es un identificador) y `TotalCharges` (figura como texto y tiene 11 valores vacíos). `Churn` se convirtió en 1 (el cliente se fue) y 0 (se quedó). Las columnas de texto se convirtieron en columnas booleanas, quitando la primera categoría de cada una como referencia.
3. **División:** 80% para entrenamiento (5.634 clientes) y 20% para test (1.409 clientes), con `random_state=36`.
4. **Modelo:** `LogisticRegression` con `max_iter=1000`.
5. **Evaluación:** matriz de confusión, accuracy, precision y recall sobre el test.
6. **Interpretación:** se leyó el signo de cada coeficiente, que indica si la variable se asocia con más o menos probabilidad de que el cliente se vaya.


## Hallazgos

### Regresión lineal

- **RMSE de 68.754:** en el test, el modelo se equivoca en promedio unos 68.754 dólares (se asume que el precio está en dólares; el dataset no lo declara). Sin una referencia, no se puede decir si es mucho o poco.
- **Ingreso mediano:** cada unidad más de `median_income` se asocia con unos 38.654 más de precio predicho, con las demás variables iguales.
- **Ubicación:** respecto a `INLAND`, las zonas `NEAR OCEAN`, `<1H OCEAN` y `NEAR BAY` tienen un precio predicho más alto (unos 45.280, 39.473 y 35.800). `ISLAND` tiene el coeficiente más alto (196.012), pero son solo 5 zonas, por lo que es poco firme.
- **Tope en el precio:** 500.001 se repite en 965 zonas, por lo que el modelo no puede aprender cuánto valen realmente.


### Regresión logística

- **Accuracy 0,78, precision 0,65 y recall 0,52** en el test.
- **El accuracy engaña:** de los 1.409 clientes del test, 1.004 se quedaron, así que un modelo que siempre dijera "se queda" acertaría en esos 1.004 casos sin detectar a ningún cliente que se va.
- **Lo más flojo es el recall:** el modelo no detectó a 195 de los 405 clientes que se fueron.
- **Coeficientes:** la fibra óptica (respecto a `DSL`) se asocia con más probabilidad de irse, y los contratos de uno o dos años (respecto al mensual) y más antigüedad (`tenure`), se asocian con menos. Son asociaciones, no causas.


## Tecnologías

- Python
- pandas
- scikit-learn
- Jupyter Notebook


## Estructura del repositorio

```
.
├── oraculo_inmobiliario.ipynb               # Notebook con los dos modelos
├── housing.csv                              # Dataset de California Housing Prices
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Dataset de Telco Customer Churn
├── requirements.txt                         # Dependencias
└── README.md
```

## Cómo correrlo

1. Cloná el repositorio y entrá a la carpeta:

```bash
git clone https://github.com/CarlosScura/oraculo-inmobiliario-de-california.git
cd oraculo-inmobiliario-de-california
```

2. Creá y activá un entorno virtual (en Linux y macOS):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instalá las dependencias:

```bash
pip install -r requirements.txt
```

4. Abrí el notebook y ejecutá las celdas en orden:

```bash
jupyter notebook oraculo_inmobiliario.ipynb
```

Los dos archivos `.csv` tienen que estar en la misma carpeta que el notebook.


## Autor

Carlos Alarcón Scura — Estudiante de Penguin Academy.

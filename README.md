# InterLub - Sistema de Predicción y Recomendación de Lubricantes

Este proyecto tiene como objetivo desarrollar un sistema de análisis de datos para InterLub, una empresa del sector de lubricantes. El sistema se enfoca en dos áreas principales: la predicción de propiedades de lubricantes y la recomendación de productos a clientes.

## Características

*   **Predicción de Propiedades de Lubricantes**: Se utiliza un modelo de regresión lineal para predecir propiedades importantes de los lubricantes.
*   **Aumento de Datos (Data Augmentation)**: Para mejorar la precisión del modelo de regresión, se utilizan técnicas avanzadas de simulación de datos para aumentar el tamaño del conjunto de datos. Las técnicas implementadas son:
    *   k-Nearest Neighbors (k-NN)
    *   Cópulas Gaussianas
    *   Autoencoder Variacional (VAE) con PyTorch
*   **Recomendación de Productos**: Se implementa un sistema de recomendación basado en la similitud del coseno para encontrar productos similares a partir de las características de los lubricantes.
*   **Interfaz Interactiva**: Se proporciona una aplicación web desarrollada con Streamlit para interactuar con los modelos de predicción y recomendación.

## Estructura del Proyecto

```
.
├── data/
│   ├── dataset_simulado.csv
│   └── datos_onehotenocder.csv
├── notebooks/
│   ├── 2RetoInterlub_M2003B_student.ipynb  # Notebook principal con simulación y regresión
│   ├── app.py                             # Aplicación web con Streamlit
│   ├── Cos_Model.py                       # Modelo de similitud del coseno
│   ├── Datos.ipynb                        # Preprocesamiento y codificación de datos
│   └── Simulation_and_Regression_Functions.py # Funciones de simulación y regresión
├── scripts/
│   └── Cos_Model.py                       # Script del modelo de similitud
├── requirements.txt                       # Dependencias del proyecto
└── README.md                              # Este archivo
```

## Instalación

1.  Clona este repositorio:
    ```bash
    git clone <URL-DEL-REPOSITORIO>
    ```
2.  Crea un entorno virtual e instálalo:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Windows usa `.venv\Scripts\activate`
    ```
3.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

### 1. Preprocesamiento de Datos

Ejecuta el notebook `notebooks/Datos.ipynb` para realizar el preprocesamiento de los datos, incluida la codificación "one-hot" de las variables categóricas. Esto generará el archivo `data/datos_onehotenocder.csv`.

### 2. Simulación de Datos y Entrenamiento del Modelo

Ejecuta el notebook `notebooks/2RetoInterlub_M2003B_student.ipynb` para:
1.  Realizar el aumento de datos utilizando las diferentes técnicas de simulación.
2.  Entrenar el modelo de regresión lineal con los datos simulados.
3.  Evaluar el rendimiento del modelo.

### 3. Ejecutar la Aplicación Web

Para iniciar la aplicación interactiva de Streamlit, ejecuta el siguiente comando en tu terminal:

```bash
streamlit run notebooks/app.py
```

Esto abrirá una nueva pestaña en tu navegador con la aplicación, donde podrás interactuar con los modelos de predicción y recomendación.

## Modelos y Métodos

### Simulación de Datos

*   **k-NN**: Genera nuevos puntos de datos sintéticos basados en los vecinos más cercanos de los puntos existentes.
*   **Cópulas Gaussianas**: Modela la estructura de dependencia entre las variables para generar muestras sintéticas.
*   **Autoencoder Variacional (VAE)**: Una red neuronal generativa que aprende la distribución de los datos y puede generar nuevas muestras a partir de ella.

### Modelos de Machine Learning

*   **Regresión Lineal**: Se utiliza para predecir las propiedades de los lubricantes a partir de sus características.
*   **Similitud del Coseno**: Mide la similitud entre dos vectores de características de productos, permitiendo encontrar los productos más similares a uno dado.

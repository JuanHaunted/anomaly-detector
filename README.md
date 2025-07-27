# Detector de Transacciones Fraudulentas con Modelos Incrementales y Análsis Exploratorio de Datos

Para instalar las dependencias se debe clonar el repositorio y correr en consola. 
Esto creará un ambiente virtual para el proyecto con todas las dependencias necesarias.

```bash
# Instalar dependencias
pipenv install
```

```bash
# Syncronizar
pipenv sync
```

Para ejecutar la api, ir a api/main.py y ejecutar el código. 

```bash
# Ejecutar API
python main.py
```

También puede correr manualmente utilizando comando de uvicorn.

```bash
# Ejecutar API
uvicorn main:app --reload
```

## Corta descripción de cada uno de las carpetas del proyecto

### API

- main.py: Contiene la API para acceder a los modelos. 
    1. POST: predict_simple -> Predice utilizando árbol de decisión
    2. POST: predict -> Predice utilizando RandomForest Incremental
    3. POST: update -> Predice utilizando RandomForest Incremental  y Actualiza BD

### DATA

- Contiene los datos crudos brindados para resolver el problema.

### STREAM_DATA

- Contiene el split de datos procesados para el modelo incremental
- El entrenamiento de este modelo es uno a uno.

### DB

- db_connector.py: Contiene objeto genérico para comunicarse con la base de datos. 
- initialize.py: Inicializa la base de datos utilizando los datos procesados durante el entrenamiento del
modelo incremental.

### Experiments

- exploration.ipynb: Contine análsis estadístico descriptivo de los datos y algunas visualizaciones para entender
mejor el dataset.

### SRC

- pre_process.py: Contiene rutinas de preprocesado para preparar el dataset del modelo incremental. También entrena la normalización.
- simple_model.py: Contiene un modelo simple de árbol de decisión basado en la la variable 'amount'
- train.py: Contiene entrenamiento de modelo incremental utilizando RandomForest
- utilities.py: Contiene funciónes de ayuda para adaptar los datos.

### TODO

TODO: Añadir modulo de tests unitarios para el código.
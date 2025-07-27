import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from river.tree import HoeffdingAdaptiveTreeClassifier
from river import metrics
from river import evaluate
from river.metrics import ClassificationReport

# Cargamos el dataset transformado con el PowerTransformer
dataset = pd.read_csv('data/data_transactions_power_transformed.csv')

def train_incremental_model(dataset: pd.DataFrame) -> None: 
    feature_names = ['norm_amount', 'balance_ratio', 'amount_to_balance_ratio', 'transaction_type']


    train_data, test_stream_data = train_test_split(
        dataset, test_size=0.2, random_state=42, stratify=dataset['is_anomaly']
        )
    
    # Queremos guardar el stream data para poder probar el aprendizaje incremental a través de la API
    train_data.to_csv('stream_data/train_data.csv', index=False)
    test_stream_data.to_csv('stream_data/test_stream_data.csv', index=False)

    # Creamos un random forest incremental AdaptativeRandomForestClassifier
    model = HoeffdingAdaptiveTreeClassifier(
        grace_period=100,
        delta=1e-5,
        seed=42
    )

    dataset = dataset[feature_names + ['is_anomaly']]


    metrics = ClassificationReport()
    for _, row in train_data.iterrows():
        x = {feature: row[feature] for feature in feature_names}
        y = bool(row['is_anomaly'])
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

                # Update the metrics with the prediction
        if y_pred is not None:
            metrics.update(y, y_pred)

        # Now, learn from the data
        model.learn_one(x, y)


    print(f"Modelo entrenado con {len(train_data)} transacciones.")
    print("\n--- Reporte de rendimiento del modelo ---")
    print(metrics)

    # Guardamos el modelo entrenado 
    joblib.dump(model, 'models/incremental_random_forest_model.pkl')



if __name__ == "__main__":
    train_incremental_model(dataset)

import pandas as pd
import joblib
import sys
import os
from typing import Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utilities import add_feature_engineering_columns
from db.db_connector import SQLiteConnector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

SIMPLE_MODEL_PATH = 'models/decision_tree_amount_reference_model.pkl'
INCREMENTAL_MODEL_PATH = 'models/incremental_random_forest_model.pkl'
POWER_TRANSFORMER_PATH = 'models/power_transformer.pkl'
simple_model = None
incremental_model = None


# Para validación de datos
class Transaction(BaseModel):
    user_id: str
    transaction_type: str
    amount: float
    balance_before: float
    balance_after: float
    timestamp: str

class TransactionTrueLabel(BaseModel):
    user_id: str
    transaction_type: str
    amount: float
    balance_before: float
    balance_after: float
    timestamp: str
    is_anomaly: bool

# Inicializamos los modelos que hemos creado
@asynccontextmanager
async def lifespan(app: FastAPI):
    global simple_model, incremental_model, power_transformer, db_connector
    try:
        simple_model = joblib.load(SIMPLE_MODEL_PATH)
        incremental_model = joblib.load(INCREMENTAL_MODEL_PATH)
        power_transformer = joblib.load(POWER_TRANSFORMER_PATH)
        db_connector = SQLiteConnector(db_path='db/transactions.db')
        yield
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")
    

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "API de detección de fraudes corriendo correctamente."}

@app.post("/predict_simple")
def predict_simple(transaction: Transaction):
    try:
        # Realizamos la predicción con el modelo simple
        prediction = simple_model.predict([[transaction.amount]])
        confidence = simple_model.predict_proba([[transaction.amount]])[0][1]  # Probabilidad de ser fraudulenta

        # El modelo retorna la predicción y que tan confiado está de que sea fraudulenta o no
        return {"prediction": int(prediction[0]), "confidence": f"{confidence:.2%}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
    

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Preparamos los datos de la transacción para la predicción
        transaction_df = pd.DataFrame(transaction.model_dump(), index=[0])
        transaction_df = add_feature_engineering_columns(transaction_df)
        features_to_transform = ['norm_amount', 'balance_ratio', 'amount_to_balance_ratio']
        feature_names = ['norm_amount', 'balance_ratio', 'amount_to_balance_ratio', 'transaction_type']
        transaction_df[features_to_transform] = power_transformer.transform(transaction_df[features_to_transform])
        X = transaction_df[feature_names].iloc[0].to_dict()

        # Realizamos la predicción
        prediction_proba = incremental_model.predict_proba_one(X)
        prediction = max(prediction_proba, key=prediction_proba.get)
        confidence = prediction_proba.get(True, 0.0)


        return {"prediction": 1 if prediction else 0,
        "confidence": f"{confidence:.2%}",
        "model_updated": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
    

# Endpoint para realizar la predicción y actualizar el modelo
@app.post("/update")
def predict(transaction: Union[Transaction, TransactionTrueLabel], add_to_db: bool = True):
    has_label = False
    if hasattr(transaction, 'is_anomaly'):
        has_label = True

    try:
        # Preparamos los datos de la transacción para la predicción
        transaction_df = pd.DataFrame(transaction.model_dump(), index=[0])
        transaction_df = add_feature_engineering_columns(transaction_df)
        features_to_transform = ['norm_amount', 'balance_ratio', 'amount_to_balance_ratio']
        feature_names = ['norm_amount', 'balance_ratio', 'amount_to_balance_ratio', 'transaction_type']
        transaction_df[features_to_transform] = power_transformer.transform(transaction_df[features_to_transform])
        X = transaction_df[feature_names].iloc[0].to_dict()

        # Realizamos la predicción
        prediction_proba = incremental_model.predict_proba_one(X)
        prediction = max(prediction_proba, key=prediction_proba.get)
        confidence = prediction_proba.get(True, 0.0)


        prediction = 1 if prediction else 0

        # Actualizamos el modelo incremental con la nueva transacción (utilizando el label verdadero)
        if has_label:
            incremental_model.learn_one(X, transaction.is_anomaly)
        else:
            incremental_model.learn_one(X, prediction)

        if add_to_db:
            # Guardamos la transacción en la base de datos
            db_connector.insert_record(transaction_df, 'transactions')

        return {"prediction": prediction,
        "confidence": f"{confidence:.2%}",
        "model_updated": True,
        "matched": prediction == transaction.is_anomaly,
        "added_to_db": add_to_db
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
    
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
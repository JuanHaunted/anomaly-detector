import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_text
import joblib

# Cómo sabemos que la variable amount tiene una correlación muy alta con la variable is_fraudulent,
# vamos a entrenar un árbol de decisión con una sola partición para tener como modelo de referencia. 
# Nuestro objetivo será superar este modelo de referencia con el modelo incremental.

def train_model() -> None:
    dataset = pd.read_csv('data/data_transactions.csv')

    X = dataset[['amount']]
    Y = dataset['is_anomaly']

    # Stratify en el target porque solo ~1% de las transacciones son fraudulentas
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))

    # Ahora guardamos el modelo de referencia
    joblib.dump(model, 'models/decision_tree_amount_reference_model.pkl')

    tree_rules = export_text(model, feature_names=["amount"])
    print(tree_rules)

if __name__ == "__main__":
    train_model()
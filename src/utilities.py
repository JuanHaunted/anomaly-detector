import pandas as pd

def add_feature_engineering_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Adds feature engineering columns to the dataset.
    
    Parameters:
    dataset (pd.DataFrame): The input dataset.
    
    Returns:
    pd.DataFrame: The dataset with new feature engineering columns.
    """
    dataset_eng = dataset.copy()
    
    # Adding balance ratio
    dataset_eng['balance_ratio'] = dataset_eng['balance_after'] / (dataset_eng['balance_before'] + 1e-8)
    
    # Adding amount to balance ratio
    dataset_eng['amount_to_balance_ratio'] = dataset_eng['amount'] / (dataset_eng['balance_before'] + 1e-8)

    dataset_eng['norm_amount'] = dataset_eng['amount']

    dataset_eng['transaction_type'] = dataset_eng['transaction_type'].map({
    'recarga': 1,
    'retiro': -1
    })

    
    return dataset_eng
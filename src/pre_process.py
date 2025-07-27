import numpy as np
from sklearn.preprocessing import PowerTransformer
import pandas as pd
from utilities import add_feature_engineering_columns
import joblib

# As we saw in the experiments, the data is very skewed
# Therefore, it is idea to use a normalization technique such as PowerTransformer
# to normalize the data.

#load dataset
dataset = pd.read_csv('data/data_transactions.csv')

dataset_power = add_feature_engineering_columns(dataset.copy())


# Initialize the power transformer
power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
features_to_transform = ['norm_amount', 'balance_ratio', 'amount_to_balance_ratio']

# Fit the power transformer on the numerical columns
dataset_power[features_to_transform] = power_transformer.fit_transform(dataset_power[features_to_transform])

joblib.dump(power_transformer, 'models/power_transformer.pkl')

# Save the transformed dataset
dataset_power.to_csv('data/data_transactions_power_transformed.csv', index=False)



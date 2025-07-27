import pandas as pd
from db_connector import SQLiteConnector
import os

sqlconnector = SQLiteConnector('db/transactions.db')

sqlconnector.initialize_db_from_csv('./stream_data/train_data.csv', 'transactions')


import sqlite3
import pandas as pd
from typing import Optional, Dict

class SQLiteConnector:
    def __init__(self, db_path: str = 'transactions.db'):
        self.db_path = db_path

    def initialize_db_from_csv(self, csv_path: str, table_name: str) -> None:
        df = pd.read_csv(csv_path)
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)

    def insert_record(self, record: Dict, table_name: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            #Process data
            pd_record = pd.DataFrame([record])
            query = f"INSERT INTO {table_name} ({', '.join(pd_record.columns)}) VALUES ({', '.join(['?' for _ in pd_record.columns])})"
            cursor.execute(query, tuple(pd_record.iloc[0]))

    def fetch_all_records(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            query = f"SELECT * FROM {table_name}"
            if limit is not None:
                query += f" LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
        return df
    
    def excute_query(self, query: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
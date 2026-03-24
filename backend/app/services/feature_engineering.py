import pandas as pd
import numpy as np


def create_features(df, balance_col, withdraw_col, deposit_col):
    """Engineer numeric features for the Isolation Forest model."""

    df = df.copy()

    # Net amount
    df['Amount'] = df[deposit_col] - df[withdraw_col]

    # Date-based features
    date_col = next((c for c in df.columns if 'DATE' in c.upper()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['Day']   = df[date_col].dt.day.fillna(0).astype(int)
        df['Month'] = df[date_col].dt.month.fillna(0).astype(int)
        df['Hour']  = df[date_col].dt.hour.fillna(0).astype(int)
    else:
        df['Day']   = 0
        df['Month'] = 0
        df['Hour']  = 0

    features = [withdraw_col, deposit_col, balance_col, 'Amount', 'Day', 'Month', 'Hour']

    # Drop rows where ALL feature columns are NaN, then fill remaining NaN with 0
    df[features] = df[features].fillna(0)

    return df, features

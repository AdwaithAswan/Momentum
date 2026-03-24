import pandas as pd


def preprocess(df):
    """Clean and normalise the uploaded CSV.

    Returns (df, balance_col, withdraw_col, deposit_col).
    Falls back gracefully when expected columns are absent.
    """
    # Normalise column names: strip whitespace
    df.columns = df.columns.str.strip()

    def find_col(df, keywords):
        """Return first column whose upper-cased name contains any keyword."""
        for kw in keywords:
            for c in df.columns:
                if kw in c.upper():
                    return c
        return None

    balance_col  = find_col(df, ['BALANCE'])
    withdraw_col = find_col(df, ['WITHDRAWAL', 'DEBIT', 'WITHDRAW'])
    deposit_col  = find_col(df, ['DEPOSIT', 'CREDIT'])

    # If expected money columns are absent, create synthetic ones from 'amount'
    if not withdraw_col:
        amount_col = find_col(df, ['AMOUNT'])
        if amount_col:
            df['Withdrawal Amt'] = df[amount_col].clip(lower=0)
            df['Deposit Amt']    = 0
            withdraw_col = 'Withdrawal Amt'
            deposit_col  = 'Deposit Amt'
        else:
            df['Withdrawal Amt'] = 0
            df['Deposit Amt']    = 0
            withdraw_col = 'Withdrawal Amt'
            deposit_col  = 'Deposit Amt'

    if not deposit_col:
        df['Deposit Amt'] = 0
        deposit_col = 'Deposit Amt'

    if not balance_col:
        df['Balance Amt'] = 0
        balance_col = 'Balance Amt'

    # Clean numeric columns
    for col in [balance_col, withdraw_col, deposit_col]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace(' ', '', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df, balance_col, withdraw_col, deposit_col

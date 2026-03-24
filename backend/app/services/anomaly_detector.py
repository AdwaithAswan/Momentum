import numpy as np
import pandas as pd
from .model import train_model


def detect_anomalies(df, X):
    """Run IsolationForest on X adds score risk anomaly type."""

    model, X_vals = train_model(X)   

    df = df.copy()

    
    raw_scores        = model.decision_function(X_vals)
    df['Score']       = -raw_scores

    min_s, max_s = df['Score'].min(), df['Score'].max()
    df['Score']   = (df['Score'] - min_s) / (max_s - min_s) if max_s > min_s else 0.0

    predictions   = model.predict(X_vals)
    df['Anomaly'] = np.where(predictions == -1, 1, 0)

    
    df['Risk'] = pd.cut(
        df['Score'],
        bins=[-0.001, 0.50, 0.75, 1.001],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    ).astype(str)

    df['anomaly_type'] = ''

    flagged = df['Anomaly'] == 1
    if flagged.sum() == 0:
        return df   

    
    amount_col = next((c for c in df.columns if c == 'Amount'), None)
    if amount_col is None:
        amount_col = next((c for c in df.columns if 'AMOUNT' in c.upper()), None)

    date_col = next((c for c in df.columns if 'DATE' in c.upper()), None)

    acct_col = next((c for c in df.columns
                     if any(k in c.upper() for k in ['ACCOUNT', 'ACCT', 'ACC_ID'])), None)

    
    fi = df[flagged].index

    
    if amount_col:
        abs_amt = df.loc[fi, amount_col].abs().astype(float)
        median_amt = df[amount_col].abs().median()
    else:
        abs_amt    = pd.Series(0.0, index=fi)
        median_amt = 1.0

   
    label_masks = {}

    # 1. Unusually Large Transaction
    if amount_col and median_amt > 0:
        label_masks['Unusually Large Transaction'] = abs_amt > 3 * median_amt

    # 2. Rounded Amount
    if amount_col:
        label_masks['Rounded Amount'] = (
            ((abs_amt >= 1000) & (abs_amt % 5000 == 0)) |
            ((abs_amt >= 500)  & (abs_amt % 1000 == 0))
        )

    # 3. Split Transaction — just below reporting threshold
    if amount_col:
        thresholds = [10000, 50000, 100000, 200000, 500000]
        split_mask = pd.Series(False, index=fi)
        for t in thresholds:
            split_mask |= (abs_amt >= t * 0.90) & (abs_amt < t)
        label_masks['Split Transaction'] = split_mask

    # 4. High Frequency / Rapid Fund Transfer — count same-account txns per day
    if acct_col and date_col:
        try:
            dates_all  = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
            acct_date  = df[acct_col].astype(str) + '|' + dates_all.astype(str)
            daily_freq = acct_date.map(acct_date.value_counts())
            label_masks['High Frequency']      = (daily_freq.loc[fi] >= 5)
            label_masks['Rapid Fund Transfer'] = (
                (daily_freq.loc[fi] >= 3) & (daily_freq.loc[fi] < 5)
            )
        except Exception:
            pass

    # 5. Dormant Account Activity — account has very few total transactions
    if acct_col:
        acct_freq = df[acct_col].map(df[acct_col].value_counts())
        label_masks['Dormant Account Activity'] = acct_freq.loc[fi] <= 2

    # 6. Near Duplicate / Repeated Vendor Payment
    if acct_col and amount_col:
        acct_amt_key  = df[acct_col].astype(str) + '|' + df[amount_col].abs().astype(str)
        dup_count     = acct_amt_key.map(acct_amt_key.value_counts())
        label_masks['Near Duplicate']          = dup_count.loc[fi] >= 3
        label_masks['Repeated Vendor Payment'] = (
            (dup_count.loc[fi] == 2)
        )

    # 7. Sudden Behaviour Change — borderline score, no other label matched yet
    label_masks['Sudden Behaviour Change'] = df.loc[fi, 'Score'] <= 0.55

    
    
    label_df = pd.DataFrame({k: v.reindex(fi).fillna(False)
                              for k, v in label_masks.items()})

    
    def row_labels(row):
        matched = [col for col, val in row.items() if val]
        # Remove lower-priority labels when higher ones already match
        if 'Rapid Fund Transfer' in matched and 'High Frequency' in matched:
            matched.remove('Rapid Fund Transfer')
        if 'Repeated Vendor Payment' in matched and 'Near Duplicate' in matched:
            matched.remove('Repeated Vendor Payment')
        if 'Sudden Behaviour Change' in matched and len(matched) > 1:
            matched.remove('Sudden Behaviour Change')
        return ', '.join(matched) if matched else 'Unusual Pattern'

    df.loc[fi, 'anomaly_type'] = label_df.apply(row_labels, axis=1).values

    return df

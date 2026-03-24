import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from config import MODEL_PATH


def train_model(X):
    """Train IsolationForest, always fast regardless of input size."""

    # Convert to numpy for speed
    X_vals = X.values if hasattr(X, 'values') else np.array(X)

    # Replace any NaN/inf that slipped through
    X_vals = np.nan_to_num(X_vals, nan=0.0, posinf=0.0, neginf=0.0)

    # Cap training sample at 3000 rows — statistically sufficient for IsolationForest
    if len(X_vals) > 3000:
        idx    = np.random.choice(len(X_vals), 3000, replace=False)
        X_fit  = X_vals[idx]
    else:
        X_fit  = X_vals

    model = IsolationForest(
        n_estimators=40,    # fast — 40 trees is plenty for anomaly detection
        contamination=0.03,
        max_samples=min(256, len(X_fit)),
        random_state=42,
        n_jobs=-1,          # use all CPU cores
    )
    model.fit(X_fit)
    joblib.dump(model, MODEL_PATH)

    return model, X_vals   # return X_vals so caller can score all rows

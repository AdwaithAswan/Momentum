# Momentum — Bank Anomaly Detection

## Project Structure

```
momentum/
└── backend/
    ├── run.py              ← ONLY file you need to run
    ├── config.py
    ├── requirements.txt
    ├── static/             ← Frontend (served automatically by Flask)
    │   ├── index.html
    │   ├── dashboard.html
    │   ├── flagged.html
    │   ├── account.html
    │   ├── explorer.html
    │   ├── export.html
    │   ├── data.js
    │   └── style.css
    ├── models/
    │   └── isolation_forest.pkl
    ├── uploads/            ← auto-created
    ├── outputs/            ← auto-created
    └── app/
        ├── routes.py
        ├── services/
        │   ├── preprocess.py
        │   ├── feature_engineering.py
        │   ├── anomaly_detector.py
        │   └── model.py
        └── utils/
            └── file_handler.py
```

## How to Run (single command)

```bash
cd backend
pip install -r requirements.txt
python run.py
```

That's it. Flask starts, your browser opens at http://localhost:5000 automatically.

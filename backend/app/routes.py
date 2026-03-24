import time
import pandas as pd
from flask import Blueprint, request, jsonify

from app.services.preprocess import preprocess
from app.services.feature_engineering import create_features
from app.services.anomaly_detector import detect_anomalies
from app.utils.file_handler import save_file, save_output

bp = Blueprint('main', __name__)

#input the file
@bp.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    #starts timer to keep track of processing time
    start_ms = time.time()

    try:
        file_path = save_file(file)

        
        if file.filename.lower().endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

        total_records = len(df)

        #ml pipeline
        df, balance_col, withdraw_col, deposit_col = preprocess(df)
        df, features = create_features(df, balance_col, withdraw_col, deposit_col)
        df = detect_anomalies(df, df[features])

        processing_ms = int((time.time() - start_ms) * 1000)

        #output file
        save_output(df, "fraud_output.csv")

        
        df_out = df.fillna('')

       #required output parameters
        df_out['flag']  = df_out['Anomaly'].astype(int)
        df_out['score'] = df_out['Score']   
        df_out['risk']  = df_out['Risk']    

        if 'anomaly_type' not in df_out.columns:
            df_out['anomaly_type'] = ''

        #similar other column names
        acct_keywords = ['ACCOUNT_ID', 'ACCOUNT ID', 'ACCOUNTID', 'ACCT_ID',
                         'ACCOUNT NO', 'ACCOUNT NUMBER', 'ACCOUNT_NO',
                         'ACC_ID', 'ACCOUNT', 'ACCT']
        acct_col_found = None
        for kw in acct_keywords:
            for c in df_out.columns:
                if kw in c.upper().replace(' ', '_'):
                    acct_col_found = c
                    break
            if acct_col_found:
                break

        if acct_col_found and acct_col_found != 'account_id':
            df_out['account_id'] = df_out[acct_col_found].astype(str)
        elif 'account_id' not in df_out.columns:
            # last resort: generate synthetic account IDs from row index
            df_out['account_id'] = ['ACC-' + str(i).zfill(4) for i in range(len(df_out))]

        
        txn_keywords = ['TRANSACTION_ID', 'TRANSACTION ID', 'TXN_ID', 'TXN ID',
                        'TRANS_ID', 'TRANSACTIONID', 'ID']
        txn_col_found = None
        for kw in txn_keywords:
            for c in df_out.columns:
                if kw in c.upper().replace(' ', '_'):
                    txn_col_found = c
                    break
            if txn_col_found:
                break

        if txn_col_found and txn_col_found != 'transaction_id':
            df_out['transaction_id'] = df_out[txn_col_found].astype(str)
        elif 'transaction_id' not in df_out.columns:
            df_out['transaction_id'] = ['TXN-' + str(i).zfill(5) for i in range(len(df_out))]

        fraud_count = int(df_out['flag'].sum())

        
        flagged_rows = df_out[df_out['flag'] == 1]
        clean_rows   = df_out[df_out['flag'] == 0].head(100)
        combined     = pd.concat([flagged_rows, clean_rows])
        rows = combined.to_dict(orient='records')
        #return
        return jsonify({
            'message':       'Fraud detection complete',
            'fraud_count':   fraud_count,
            'total_records': total_records,
            'filename':      file.filename,
            'processing_ms': processing_ms,
            'data':          rows,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

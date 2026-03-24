import os
from config import UPLOAD_FOLDER, OUTPUT_FOLDER

def save_file(file):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)   # 🔥 ensures folder exists
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    return path

def save_output(df, filename):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)   # 🔥 ensures folder exists
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    df.to_csv(output_path, index=False)
    return output_path
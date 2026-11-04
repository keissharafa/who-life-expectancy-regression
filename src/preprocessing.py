import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def run_preprocessing(filepath):
    try:
        df = pd.read_csv(filepath, sep=',')
        if 'Life expectancy' not in df.columns and 'Life expectancy ' not in df.columns:
            df = pd.read_csv(filepath, sep=';')
    except Exception:
        df = pd.read_csv(filepath)

    # Membersihkan spasi ekstra pada seluruh nama kolom
    df.columns = df.columns.str.strip()
    
    print(f"[INFO] Kolom yang terdeteksi di dataset: {df.columns.tolist()}")

    # Imputasi missing values numerik dengan median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Imputasi missing values kategorikal dengan modus
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # Encoding kolom kategorikal
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    return df
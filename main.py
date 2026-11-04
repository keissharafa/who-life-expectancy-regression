import sys
import os

# Agar folder 'src' terbaca sistem
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import run_preprocessing
from src.splitter import run_splitting
from src.train import train_svr_model
from src.evaluate import evaluate_model

if __name__ == '__main__':
    print("[INFO] Menjalankan Pipeline Regresi SVR...")
    
    # 1. Preprocessing
    df = run_preprocessing('data/input/Life Expectancy Data.csv')
    
    # 2. Splitting & Scaling
    X_train_s, X_test_s, y_train, y_test = run_splitting(df)
    
    # 3. Training
    model = train_svr_model(X_train_s, y_train)
    
    # 4. Evaluasi & Visualisasi
    evaluate_model(model, X_test_s, y_test)
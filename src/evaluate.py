import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print(" HASIL EVALUASI MODEL SVR ".center(50))
    print("=" * 50)
    print(f"-> MAE  : {mae:.4f}")
    print(f"-> MSE  : {mse:.4f}")
    print(f"-> RMSE : {rmse:.4f}")
    print(f"-> R²   : {r2:.4f}")
    print("=" * 50 + "\n")
    
    # Visualisasi
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='darkorange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    plt.title('Actual vs Predicted Life Expectancy (SVR)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    
    os.makedirs('assets', exist_ok=True)
    plot_path = 'assets/plot_actual_vs_predicted.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Grafik disimpan di: {plot_path}")
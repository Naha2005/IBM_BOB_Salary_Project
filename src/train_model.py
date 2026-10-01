"""
train_model.py
--------------
Trains a Linear Regression model on the salary dataset, evaluates it,
saves the trained model to disk, and generates a scatter plot.
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import load_data, summarize


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'salary_model.pkl')
PLOT_PATH   = os.path.join(BASE_DIR, 'reports', 'regression_plot.png')
METRICS_PATH = os.path.join(BASE_DIR, 'reports', 'metrics.json')


def train_and_evaluate() -> dict:
    """
    Full pipeline: load → split → train → evaluate → save.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics and model coefficients.
    """
    # 1. Load data
    df = load_data()
    summarize(df)

    X = df[['YearsExperience']].values
    y = df['Salary'].values

    # 2. Train / test split (80 / 20, fixed seed for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[INFO] Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    # 3. Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Evaluate on test set
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n== Model Evaluation =================================")
    print(f"  R2 Score : {r2:.4f}")
    print(f"  MAE      : ${mae:,.2f}")
    print(f"  RMSE     : ${rmse:,.2f}")
    print(f"  Intercept: {model.intercept_:.2f}")
    print(f"  Slope    : {model.coef_[0]:.2f}")
    print("=====================================================\n")

    # 5. Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Model saved to: {MODEL_PATH}")

    # 6. Save metrics to JSON (used later by the HTML report generator)
    metrics = {
        "r2":        round(r2,   4),
        "mae":       round(mae,  2),
        "rmse":      round(rmse, 2),
        "intercept": round(float(model.intercept_), 2),
        "slope":     round(float(model.coef_[0]),   2),
        "n_train":   int(len(X_train)),
        "n_test":    int(len(X_test)),
        "n_total":   int(len(X)),
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    # 7. Generate regression plot
    _plot_regression(X, y, model)

    return metrics


def _plot_regression(X, y, model) -> None:
    """Save a scatter plot with the fitted regression line."""
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='steelblue', edgecolors='white', s=70, zorder=3, label='Actual data')
    plt.plot(x_line, y_line, color='tomato', linewidth=2.5, label='Regression line')
    plt.xlabel('Years of Experience', fontsize=12)
    plt.ylabel('Salary (USD)', fontsize=12)
    plt.title('Salary vs. Years of Experience — Linear Regression', fontsize=13)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"[INFO] Plot saved to: {PLOT_PATH}")


if __name__ == '__main__':
    train_and_evaluate()

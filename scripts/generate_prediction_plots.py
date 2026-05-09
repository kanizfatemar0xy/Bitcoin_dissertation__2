"""
Script: generate_prediction_plots.py
====================================
ML Prediction plots (Actual vs Predicted) + Metrics + Scatter
Research / IEEE Ready Version

Run:
    py generate_prediction_plots.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─── Paths ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "predictions")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── LOAD DATA (EDIT THIS PART) ───────────────────────────
# ✅ OPTION 1: Load from .npy
# y_test_exp2 = np.load("data/y_test_exp2.npy")
# rf_pred_exp2 = np.load("data/rf_pred_exp2.npy")

# y_test_exp3 = np.load("data/y_test_exp3.npy")
# gb_pred_exp3 = np.load("data/gb_pred_exp3.npy")

# ✅ OPTION 2: Load from CSV
# df = pd.read_csv("data/predictions_exp2.csv")
# y_test_exp2 = df["actual"]
# rf_pred_exp2 = df["predicted"]

# ─── TEMP DUMMY DATA (REMOVE THIS) ────────────────────────
np.random.seed(0)
y_test_exp2 = np.random.rand(100)
rf_pred_exp2 = y_test_exp2 + np.random.normal(0, 0.05, 100)

y_test_exp3 = np.random.rand(100)
gb_pred_exp3 = y_test_exp3 + np.random.normal(0, 0.05, 100)

# ─── Helpers ─────────────────────────────────────────────
def smooth(series, window=5):
    return pd.Series(series).rolling(window, min_periods=1).mean()

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ─── Plot Function ───────────────────────────────────────
def plot_prediction(y_true, y_pred, title, filename):
    rmse, mae, r2 = evaluate(y_true, y_pred)

    plt.figure(figsize=(12,4))

    # Raw lines
    plt.plot(y_true, label='Actual', color='gray', linewidth=1.5)
    plt.plot(y_pred, label='Predicted', color='green', linestyle='--', alpha=0.8)

    # Smoothed lines (optional but pro)
    plt.plot(smooth(y_true), color='black', linewidth=1.2, alpha=0.6)
    plt.plot(smooth(y_pred), color='green', linewidth=1.2, alpha=0.6)

    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Test Samples')
    plt.ylabel('Volatility (7d)')
    plt.legend()

    # Metrics text on plot
    plt.text(
        0.01, 0.95,
        f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    plt.grid(alpha=0.3)

    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {filename}")
    print(f"   RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

# ─── Scatter Plot (Research Must) ────────────────────────
def plot_scatter(y_true, y_pred, filename):
    plt.figure(figsize=(5,5))

    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Scatter")

    # perfect line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

    plt.grid(alpha=0.3)

    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {filename}")

# ─── Generate Plots ─────────────────────────────────────

# Figure 5 — Random Forest (Exp 2)
plot_prediction(
    y_test_exp2,
    rf_pred_exp2,
    'Actual vs Predicted — Random Forest | Exp 2 — Price + Tweets',
    'exp2_random_forest.png'
)

plot_scatter(
    y_test_exp2,
    rf_pred_exp2,
    'exp2_random_forest_scatter.png'
)

# Figure 6 — Gradient Boosting (Exp 3)
plot_prediction(
    y_test_exp3,
    gb_pred_exp3,
    'Actual vs Predicted — Gradient Boosting | Exp 3 — Price + Tweets + News',
    'exp3_gradient_boosting.png'
)

plot_scatter(
    y_test_exp3,
    gb_pred_exp3,
    'exp3_gradient_boosting_scatter.png'
)

print("\n🎯 All prediction plots saved in:")
print("   plots/predictions/")
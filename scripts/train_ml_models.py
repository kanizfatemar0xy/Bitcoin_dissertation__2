import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# ⚙️  PATHS
# ══════════════════════════════════════════════════════════════
BASE        = r"C:\Users\HAROON KHAN\Desktop\bitcoin_volatility_project"
DATA        = BASE + r"\data"
MODELS_DIR  = BASE + r"\models"
PLOTS_DIR   = BASE + r"\plots\ml_results"
MASTER_FILE = DATA + r"\master_dataset.csv"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
# ══════════════════════════════════════════════════════════════

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black",   "font.family": "DejaVu Sans",
    "axes.titlesize": 12,        "axes.titleweight": "bold",
    "axes.labelsize": 10,        "xtick.labelsize": 9,
    "ytick.labelsize": 9,        "legend.fontsize": 9,
    "grid.color": "#cccccc",     "grid.linestyle": "--",
    "grid.alpha": 0.5,
})

# ══════════════════════════════════════════════════════════════
# STEP 1: Load & Prepare
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Master Dataset")
print("=" * 60)

df = pd.read_csv(MASTER_FILE, parse_dates=["Date"])
print(f"  Shape      : {df.shape}")
print(f"  Date range : {df['Date'].min().date()} --> {df['Date'].max().date()}")

# Features & Target
FEATURES = [
    "Close", "Open", "High", "Low", "Volume",
    "Daily_Return", "Volatility_30d",
    "Tweet_Sentiment", "Tweet_Count", "Tweet_Positive", "Tweet_Negative",
    "News_Sentiment",  "News_Count",  "News_Positive",  "News_Negative",
    "Close_Lag1", "Close_Lag2", "Close_Lag3", "Close_Lag7",
    "Return_Lag1", "Return_Lag2", "Return_Lag3", "Return_Lag7",
    "Tweet_Sentiment_Lag1", "Tweet_Sentiment_Lag3", "Tweet_Sentiment_Lag7",
    "News_Sentiment_Lag1",  "News_Sentiment_Lag3",  "News_Sentiment_Lag7",
    "Close_MA7", "Close_MA30", "Return_Std7",
    "Tweet_Sent_MA7", "News_Sent_MA7",
    "Volume_MA7", "Price_Range", "Price_Range_Pct",
]
TARGET = "Volatility_7d"

X = df[FEATURES].values
y = df[TARGET].values
dates = df["Date"].values

print(f"  Features   : {len(FEATURES)}")
print(f"  Target     : {TARGET}")

# ══════════════════════════════════════════════════════════════
# STEP 2: Train/Test Split (Time-Series Aware — NO shuffle)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Train/Test Split (80/20)")
print("=" * 60)

split = int(len(X) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test       = dates[split:]

print(f"  Train : {len(X_train)} rows")
print(f"  Test  : {len(X_test)}  rows")

# ══════════════════════════════════════════════════════════════
# STEP 3: Scaler
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Fitting Scaler (MinMaxScaler)")
print("=" * 60)

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("  ✅ Scaler fitted & saved: models/scaler.pkl")

# ══════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  📊 {name} Results:")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     MAE  : {mae:.4f}")
    print(f"     R²   : {r2:.4f}")
    return {"Model": name, "RMSE": round(rmse,4),
            "MAE": round(mae,4), "R2": round(r2,4)}

def save_plot(name, dates, y_true, y_pred, filename):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, y_true,  color="black", linewidth=1.2, label="Actual Volatility")
    ax.plot(dates, y_pred,  color="gray",  linewidth=1.2,
            linestyle="--", label="Predicted Volatility")
    ax.set_title(f"{name} — Actual vs Predicted Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot saved: plots/ml_results/{filename}")

results = []

# ══════════════════════════════════════════════════════════════
# STEP 4: Random Forest
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Random Forest")
print("=" * 60)

print("  Training...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)
rf_pred = rf.predict(X_test_sc)
res_rf  = evaluate("Random Forest", y_test, rf_pred)
results.append(res_rf)

print("\n  Fine-tuning with GridSearchCV...")
rf_params = {
    "n_estimators": [100, 200],
    "max_depth":    [5, 10, None],
    "min_samples_split": [2, 5],
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                       rf_params, cv=3, scoring="neg_rmse"
                       if hasattr(GridSearchCV, 'neg_rmse') else "neg_mean_squared_error",
                       n_jobs=-1)
rf_grid.fit(X_train_sc, y_train)
rf_best      = rf_grid.best_estimator_
rf_best_pred = rf_best.predict(X_test_sc)
res_rf_tuned = evaluate("Random Forest (Tuned)", y_test, rf_best_pred)
results.append(res_rf_tuned)
print(f"  Best params: {rf_grid.best_params_}")

joblib.dump(rf_best, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
print("  ✅ Saved: models/random_forest_model.pkl")
save_plot("Random Forest", dates_test, y_test, rf_best_pred, "rf_actual_vs_pred.png")

# ══════════════════════════════════════════════════════════════
# STEP 5: XGBoost
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: XGBoost")
print("=" * 60)

print("  Training...")
xgb = XGBRegressor(n_estimators=100, random_state=42,
                   verbosity=0, eval_metric="rmse")
xgb.fit(X_train_sc, y_train)
xgb_pred = xgb.predict(X_test_sc)
res_xgb  = evaluate("XGBoost", y_test, xgb_pred)
results.append(res_xgb)

print("\n  Fine-tuning...")
xgb_params = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
}
xgb_grid = GridSearchCV(XGBRegressor(random_state=42, verbosity=0,
                                      eval_metric="rmse"),
                        xgb_params, cv=3,
                        scoring="neg_mean_squared_error", n_jobs=-1)
xgb_grid.fit(X_train_sc, y_train)
xgb_best      = xgb_grid.best_estimator_
xgb_best_pred = xgb_best.predict(X_test_sc)
res_xgb_tuned = evaluate("XGBoost (Tuned)", y_test, xgb_best_pred)
results.append(res_xgb_tuned)
print(f"  Best params: {xgb_grid.best_params_}")

joblib.dump(xgb_best, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
print("  ✅ Saved: models/xgboost_model.pkl")
save_plot("XGBoost", dates_test, y_test, xgb_best_pred, "xgb_actual_vs_pred.png")

# ══════════════════════════════════════════════════════════════
# STEP 6: Gradient Boosting
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Gradient Boosting")
print("=" * 60)

print("  Training...")
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train_sc, y_train)
gb_pred = gb.predict(X_test_sc)
res_gb  = evaluate("Gradient Boosting", y_test, gb_pred)
results.append(res_gb)

print("\n  Fine-tuning...")
gb_params = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 5],
    "learning_rate": [0.05, 0.1],
}
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42),
                       gb_params, cv=3,
                       scoring="neg_mean_squared_error", n_jobs=-1)
gb_grid.fit(X_train_sc, y_train)
gb_best      = gb_grid.best_estimator_
gb_best_pred = gb_grid.best_estimator_.predict(X_test_sc)
res_gb_tuned = evaluate("Gradient Boosting (Tuned)", y_test, gb_best_pred)
results.append(res_gb_tuned)
print(f"  Best params: {gb_grid.best_params_}")

joblib.dump(gb_best, os.path.join(MODELS_DIR, "gradient_boosting_model.pkl"))
print("  ✅ Saved: models/gradient_boosting_model.pkl")
save_plot("Gradient Boosting", dates_test, y_test, gb_best_pred, "gb_actual_vs_pred.png")

# ══════════════════════════════════════════════════════════════
# STEP 7: SVR
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: SVR")
print("=" * 60)

print("  Training...")
svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
svr.fit(X_train_sc, y_train)
svr_pred = svr.predict(X_test_sc)
res_svr  = evaluate("SVR", y_test, svr_pred)
results.append(res_svr)

print("\n  Fine-tuning...")
svr_params = {
    "C":       [0.1, 1.0, 10.0],
    "epsilon": [0.05, 0.1, 0.2],
    "kernel":  ["rbf", "linear"],
}
svr_grid = GridSearchCV(SVR(), svr_params, cv=3,
                        scoring="neg_mean_squared_error", n_jobs=-1)
svr_grid.fit(X_train_sc, y_train)
svr_best      = svr_grid.best_estimator_
svr_best_pred = svr_best.predict(X_test_sc)
res_svr_tuned = evaluate("SVR (Tuned)", y_test, svr_best_pred)
results.append(res_svr_tuned)
print(f"  Best params: {svr_grid.best_params_}")

joblib.dump(svr_best, os.path.join(MODELS_DIR, "svr_model.pkl"))
print("  ✅ Saved: models/svr_model.pkl")
save_plot("SVR", dates_test, y_test, svr_best_pred, "svr_actual_vs_pred.png")

# ══════════════════════════════════════════════════════════════
# STEP 8: Comparison Table + Plot
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Model Comparison")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(DATA, "ml_results.csv"), index=False)

print("\n  📊 All Models Comparison:")
print(results_df.to_string(index=False))

# Tuned only comparison
tuned = results_df[results_df["Model"].str.contains("Tuned")].copy()

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("ML Models Comparison (Tuned)", fontsize=13, fontweight="bold")

metrics = ["RMSE", "MAE", "R2"]
titles  = ["RMSE (lower is better)", "MAE (lower is better)", "R² (higher is better)"]

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i]
    bars = ax.bar(tuned["Model"].str.replace(" (Tuned)", "", regex=False),
                  tuned[metric], color="gray", edgecolor="black", width=0.5)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticklabels(
        tuned["Model"].str.replace(" (Tuned)", "", regex=False),
        rotation=20, ha="right"
    )
    for bar, val in zip(bars, tuned[metric]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ml_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✅ Saved: plots/ml_results/ml_comparison.png")

# ── Final Summary ─────────────────────────────────────────────
best = tuned.loc[tuned["RMSE"].idxmin(), "Model"]
print("\n" + "=" * 60)
print("FINAL SUMMARY — ML MODELS")
print("=" * 60)
print(f"""
  Models Trained   : Random Forest, XGBoost, Gradient Boosting, SVR
  Scaler           : MinMaxScaler (saved)
  Fine-tuning      : GridSearchCV (cv=3)
  Evaluation       : RMSE, MAE, R²

  Best ML Model    : {best.replace(' (Tuned)','')}

  Saved Models:
    models/scaler.pkl
    models/random_forest_model.pkl
    models/xgboost_model.pkl
    models/gradient_boosting_model.pkl
    models/svr_model.pkl

  Next Step → Deep Learning (LSTM + NN)
""")

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import shap
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# ⚙️  PATHS
# ══════════════════════════════════════════════════════════════
BASE        = r"C:\Users\HAROON KHAN\Desktop\bitcoin_volatility_project"
DATA        = BASE + r"\data"
MODELS_DIR  = BASE + r"\models"
PLOTS_DIR   = BASE + r"\plots\shap"
MASTER_FILE = DATA + r"\master_dataset.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)
# ══════════════════════════════════════════════════════════════

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black",   "font.family": "DejaVu Sans",
    "axes.titlesize": 12,        "axes.titleweight": "bold",
    "axes.labelsize": 10,        "xtick.labelsize": 9,
    "ytick.labelsize": 9,        "grid.color": "#cccccc",
    "grid.linestyle": "--",      "grid.alpha": 0.5,
})

# ══════════════════════════════════════════════════════════════
# STEP 1: Load Data
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Data & Models")
print("=" * 60)

df = pd.read_csv(MASTER_FILE, parse_dates=["Date"])

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

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
X_scaled = scaler.transform(X)

split      = int(len(X_scaled) * 0.80)
X_train_sc = X_scaled[:split]
X_test_sc  = X_scaled[split:]

# Feature names for plots
feat_names = FEATURES

print(f"  Data shape  : {df.shape}")
print(f"  Train/Test  : {len(X_train_sc)} / {len(X_test_sc)}")

# Load models
rf  = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
gb  = joblib.load(os.path.join(MODELS_DIR, "gradient_boosting_model.pkl"))
svr = joblib.load(os.path.join(MODELS_DIR, "svr_model.pkl"))
print("  ✅ All models loaded.")

# ══════════════════════════════════════════════════════════════
# STEP 2: SHAP — XGBoost (Best Tree Model)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: SHAP for XGBoost")
print("=" * 60)

explainer_xgb  = shap.TreeExplainer(xgb)
shap_vals_xgb  = explainer_xgb.shap_values(X_test_sc)
shap_df_xgb    = pd.DataFrame(shap_vals_xgb, columns=feat_names)

# Mean absolute SHAP
mean_shap_xgb = pd.DataFrame({
    "Feature": feat_names,
    "SHAP_Importance": np.abs(shap_vals_xgb).mean(axis=0)
}).sort_values("SHAP_Importance", ascending=False)

print("  Top 10 Features (XGBoost):")
print(mean_shap_xgb.head(10).to_string(index=False))

# Plot 1 — XGBoost Bar Summary
fig, ax = plt.subplots(figsize=(10, 7))
top10_xgb = mean_shap_xgb.head(10)
ax.barh(top10_xgb["Feature"][::-1], top10_xgb["SHAP_Importance"][::-1],
        color="gray", edgecolor="black")
ax.set_title("XGBoost — Top 10 Feature Importance (SHAP)")
ax.set_xlabel("Mean |SHAP Value|")
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_xgb_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: plots/shap/shap_xgb_bar.png")

# Plot 2 — XGBoost Beeswarm
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals_xgb, X_test_sc,
                  feature_names=feat_names, show=False, max_display=15)
plt.title("XGBoost — SHAP Summary (Beeswarm)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_xgb_beeswarm.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: plots/shap/shap_xgb_beeswarm.png")

# ══════════════════════════════════════════════════════════════
# STEP 3: SHAP — Random Forest
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: SHAP for Random Forest")
print("=" * 60)

explainer_rf = shap.TreeExplainer(rf)
shap_vals_rf = explainer_rf.shap_values(X_test_sc)
mean_shap_rf = pd.DataFrame({
    "Feature": feat_names,
    "SHAP_Importance": np.abs(shap_vals_rf).mean(axis=0)
}).sort_values("SHAP_Importance", ascending=False)

print("  Top 10 Features (Random Forest):")
print(mean_shap_rf.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
top10_rf = mean_shap_rf.head(10)
ax.barh(top10_rf["Feature"][::-1], top10_rf["SHAP_Importance"][::-1],
        color="gray", edgecolor="black")
ax.set_title("Random Forest — Top 10 Feature Importance (SHAP)")
ax.set_xlabel("Mean |SHAP Value|")
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_rf_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: plots/shap/shap_rf_bar.png")

# ══════════════════════════════════════════════════════════════
# STEP 4: SHAP — Gradient Boosting
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: SHAP for Gradient Boosting")
print("=" * 60)

explainer_gb = shap.TreeExplainer(gb)
shap_vals_gb = explainer_gb.shap_values(X_test_sc)
mean_shap_gb = pd.DataFrame({
    "Feature": feat_names,
    "SHAP_Importance": np.abs(shap_vals_gb).mean(axis=0)
}).sort_values("SHAP_Importance", ascending=False)

print("  Top 10 Features (Gradient Boosting):")
print(mean_shap_gb.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
top10_gb = mean_shap_gb.head(10)
ax.barh(top10_gb["Feature"][::-1], top10_gb["SHAP_Importance"][::-1],
        color="gray", edgecolor="black")
ax.set_title("Gradient Boosting — Top 10 Feature Importance (SHAP)")
ax.set_xlabel("Mean |SHAP Value|")
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_gb_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: plots/shap/shap_gb_bar.png")

# ══════════════════════════════════════════════════════════════
# STEP 5: SHAP — SVR (KernelExplainer — sample based)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: SHAP for SVR (KernelExplainer)")
print("=" * 60)
print("  ⏳ SVR SHAP is slow — using 50 background samples...")

background = shap.sample(X_train_sc, 50)
explainer_svr = shap.KernelExplainer(svr.predict, background)
shap_vals_svr = explainer_svr.shap_values(X_test_sc[:30], nsamples=100)

mean_shap_svr = pd.DataFrame({
    "Feature": feat_names,
    "SHAP_Importance": np.abs(shap_vals_svr).mean(axis=0)
}).sort_values("SHAP_Importance", ascending=False)

print("  Top 10 Features (SVR):")
print(mean_shap_svr.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
top10_svr = mean_shap_svr.head(10)
ax.barh(top10_svr["Feature"][::-1], top10_svr["SHAP_Importance"][::-1],
        color="gray", edgecolor="black")
ax.set_title("SVR — Top 10 Feature Importance (SHAP)")
ax.set_xlabel("Mean |SHAP Value|")
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_svr_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: plots/shap/shap_svr_bar.png")

# ══════════════════════════════════════════════════════════════
# STEP 6: Combined SHAP Comparison — All 4 ML Models
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Combined SHAP — Top Features Across Models")
print("=" * 60)

# Get top 10 from each
top_xgb = mean_shap_xgb.head(10).set_index("Feature")["SHAP_Importance"]
top_rf  = mean_shap_rf.head(10).set_index("Feature")["SHAP_Importance"]
top_gb  = mean_shap_gb.head(10).set_index("Feature")["SHAP_Importance"]
top_svr = mean_shap_svr.head(10).set_index("Feature")["SHAP_Importance"]

# Union of all top features
all_top_features = list(set(
    top_xgb.index.tolist() + top_rf.index.tolist() +
    top_gb.index.tolist()  + top_svr.index.tolist()
))

compare_df = pd.DataFrame({
    "XGBoost":          [mean_shap_xgb.set_index("Feature").loc[f, "SHAP_Importance"]
                         if f in mean_shap_xgb["Feature"].values else 0
                         for f in all_top_features],
    "Random Forest":    [mean_shap_rf.set_index("Feature").loc[f, "SHAP_Importance"]
                         if f in mean_shap_rf["Feature"].values else 0
                         for f in all_top_features],
    "Grad. Boosting":   [mean_shap_gb.set_index("Feature").loc[f, "SHAP_Importance"]
                         if f in mean_shap_gb["Feature"].values else 0
                         for f in all_top_features],
    "SVR":              [mean_shap_svr.set_index("Feature").loc[f, "SHAP_Importance"]
                         if f in mean_shap_svr["Feature"].values else 0
                         for f in all_top_features],
}, index=all_top_features)

compare_df["Mean"] = compare_df.mean(axis=1)
compare_df = compare_df.sort_values("Mean", ascending=False).head(15)
compare_df.to_csv(os.path.join(DATA, "shap_feature_importance.csv"))
print("  ✅ Saved: data/shap_feature_importance.csv")

# Grouped bar chart
fig, ax = plt.subplots(figsize=(14, 7))
x      = np.arange(len(compare_df))
width  = 0.2
colors = ["black", "#555555", "#888888", "#bbbbbb"]
models = ["XGBoost", "Random Forest", "Grad. Boosting", "SVR"]

for i, (model, color) in enumerate(zip(models, colors)):
    ax.bar(x + i*width, compare_df[model],
           width, label=model, color=color, edgecolor="black", linewidth=0.5)

ax.set_title("SHAP Feature Importance — All ML Models Comparison")
ax.set_xlabel("Features")
ax.set_ylabel("Mean |SHAP Value|")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(compare_df.index, rotation=40, ha="right", fontsize=8)
ax.legend()
ax.grid(True, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_all_models_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: plots/shap/shap_all_models_comparison.png")

# ── Final Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETE")
print("=" * 60)
print(f"""
  Saved Plots:
    plots/shap/shap_xgb_bar.png
    plots/shap/shap_xgb_beeswarm.png
    plots/shap/shap_rf_bar.png
    plots/shap/shap_gb_bar.png
    plots/shap/shap_svr_bar.png
    plots/shap/shap_all_models_comparison.png

  Saved Data:
    data/shap_feature_importance.csv

  Top Features driving Bitcoin Volatility:
""")
print(compare_df[["Mean"]].head(10).round(4).to_string())
print(f"""
  Next Step → RAG (Retrieval-Augmented Generation)
""")

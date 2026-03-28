"""
Script 6: evaluate_all_experiments.py
=======================================
Teeno experiments ke trained models ka
RMSE, MAE, R² evaluate karta hai.

Output:
  results/
  ├── exp1_results.csv
  ├── exp2_results.csv
  ├── exp3_results.csv
  ├── all_experiments_comparison.csv
  └── plots/
      ├── exp1_metrics_comparison.png
      ├── exp2_metrics_comparison.png
      ├── exp3_metrics_comparison.png
      ├── all_r2_comparison.png
      ├── all_rmse_comparison.png
      ├── all_mae_comparison.png
      └── actual_vs_predicted/
          ├── exp1_random_forest.png
          ├── exp1_xgboost.png  ... (18 plots total)

Run karo:  py evaluate_all_experiments.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
AVP_DIR     = os.path.join(PLOTS_DIR, "actual_vs_predicted")

for d in [RESULTS_DIR, PLOTS_DIR, AVP_DIR]:
    os.makedirs(d, exist_ok=True)

EXPERIMENTS = {
    "exp1": {"file": os.path.join(BASE_DIR, "master_price_news.csv"),    "label": "Exp 1 — Price + News",             "short": "Exp1"},
    "exp2": {"file": os.path.join(BASE_DIR, "master_price_tweets.csv"),  "label": "Exp 2 — Price + Tweets",           "short": "Exp2"},
    "exp3": {"file": os.path.join(BASE_DIR, "master_dataset.csv"),       "label": "Exp 3 — Price + Tweets + News",    "short": "Exp3"},
}

TARGET     = "Volatility_7d"
TEST_SPLIT = 0.20
BG         = "#FAFAFA"

MODEL_CONFIGS = [
    ("random_forest_model.pkl",     "Random Forest",     "#E67E22", "pkl"),
    ("xgboost_model.pkl",           "XGBoost",           "#2980B9", "pkl"),
    ("gradient_boosting_model.pkl", "Gradient Boosting", "#27AE60", "pkl"),
    ("svr_model.pkl",               "SVR",               "#8E44AD", "pkl"),
    ("lstm_model.keras",            "LSTM",              "#E74C3C", "keras"),
    ("nn_model.keras",              "Neural Network",    "#16A085", "keras"),
]
COLOR_MAP = {m: c for _, m, c, _ in MODEL_CONFIGS}

def load_pkl(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:  return pickle.load(f)

def make_sequences(X, y, timesteps=10):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  EVALUATE ALL EXPERIMENTS — RMSE / MAE / R²")
print("=" * 62)

all_rows = []

for exp_key, exp_info in EXPERIMENTS.items():

    print(f"\n{'─'*62}")
    print(f"  {exp_info['label']}")
    print(f"{'─'*62}")

    if not os.path.exists(exp_info["file"]):
        print(f"  ⚠  Skipped — dataset not found")
        continue

    df = pd.read_csv(exp_info["file"], parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in ["Date", TARGET]]
    X = df[feat_cols].values
    y = df[TARGET].values

    split    = int(len(X) * (1 - TEST_SPLIT))
    X_test   = X[split:]
    y_test   = y[split:]

    mdl_dir  = os.path.join(MODELS_DIR, exp_key)
    scaler_X = load_pkl(os.path.join(mdl_dir, "scaler.pkl"))
    scaler_y = load_pkl(os.path.join(mdl_dir, "y_scaler.pkl"))

    if scaler_X is None:
        print(f"  ⚠  Scaler not found — run train_all_experiments.py first!")
        continue

    X_test_sc = scaler_X.transform(X_test)
    y_test_sc = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    exp_rows = []

    for fname, mname, color, mtype in MODEL_CONFIGS:
        mpath = os.path.join(mdl_dir, fname)
        if not os.path.exists(mpath):
            print(f"  ⚠  {mname} — not found, skipping")
            continue

        try:
            if mtype == "keras":
                model = load_model(mpath)
                if "lstm" in fname:
                    TIMESTEPS = 10
                    X_seq, y_seq = make_sequences(X_test_sc, y_test_sc, TIMESTEPS)
                    pred_sc = model.predict(X_seq, verbose=0).ravel()
                    pred    = scaler_y.inverse_transform(pred_sc.reshape(-1,1)).ravel()
                    y_true  = scaler_y.inverse_transform(y_seq.reshape(-1,1)).ravel()
                else:
                    pred_sc = model.predict(X_test_sc, verbose=0).ravel()
                    pred    = scaler_y.inverse_transform(pred_sc.reshape(-1,1)).ravel()
                    y_true  = y_test
            else:
                model   = load_pkl(mpath)
                pred_sc = model.predict(X_test_sc)
                pred    = scaler_y.inverse_transform(pred_sc.reshape(-1,1)).ravel()
                y_true  = y_test

            r2   = r2_score(y_true, pred)
            rmse = np.sqrt(mean_squared_error(y_true, pred))
            mae  = mean_absolute_error(y_true, pred)

            print(f"  {mname:<22}  R²={r2:+.4f}  RMSE={rmse:.6f}  MAE={mae:.6f}")

            row = {
                "Experiment": exp_info["label"],
                "Model"     : mname,
                "R2"        : round(r2,   4),
                "RMSE"      : round(rmse, 6),
                "MAE"       : round(mae,  6),
                "Train_Rows": split,
                "Test_Rows" : len(y_true),
            }
            exp_rows.append(row)
            all_rows.append(row)

            # ── Actual vs Predicted plot ───────────────────────────
            safe = mname.lower().replace(" ", "_")
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor(BG)
            ax.plot(y_true, color="#AAAAAA", linewidth=1.0, label="Actual",    alpha=0.8)
            ax.plot(pred,   color=color,     linewidth=1.2, label="Predicted", alpha=0.9)
            style_ax(ax,
                     title  = f"Actual vs Predicted — {mname}  |  {exp_info['label']}",
                     xlabel = "Test Samples",
                     ylabel = "Volatility (7d)")
            ax.legend(fontsize=10)
            # Add metrics annotation
            ax.text(0.99, 0.97,
                    f"R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, color="#555",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
            plt.tight_layout()
            fig.savefig(os.path.join(AVP_DIR, f"{exp_key}_{safe}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            print(f"  ⚠  {mname} error: {e}")
            continue

    if not exp_rows:
        continue

    # ── Save per-experiment CSV ────────────────────────────────
    exp_df = pd.DataFrame(exp_rows)
    exp_df.to_csv(os.path.join(RESULTS_DIR, f"{exp_key}_results.csv"), index=False)
    print(f"\n  ✅ {exp_key}_results.csv saved")

    # ── Per-experiment: 3-metric bar chart ───────────────────
    models     = exp_df["Model"].tolist()
    bar_colors = [COLOR_MAP.get(m, "#888") for m in models]
    x          = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Model Metrics Comparison — {exp_info['label']}",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, metric, title in zip(
        axes,
        ["R2", "RMSE", "MAE"],
        ["R² Score (↑ higher = better)",
         "RMSE (↓ lower = better)",
         "MAE (↓ lower = better)"]
    ):
        vals = exp_df[metric].tolist()
        ax.bar(x, vals, color=bar_colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_facecolor(BG)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=7.5, color="#444")

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{exp_key}_metrics_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {exp_key}_metrics_comparison.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# COMBINED — all experiments
# ══════════════════════════════════════════════════════════════════════════════
if all_rows:
    combined_df = pd.DataFrame(all_rows)
    combined_df.to_csv(
        os.path.join(RESULTS_DIR, "all_experiments_comparison.csv"), index=False)
    print(f"\n  ✅ all_experiments_comparison.csv saved")

    model_names_all = [m for _, m, _, _ in MODEL_CONFIGS
                       if m in combined_df["Model"].values]
    exp_list = list(EXPERIMENTS.items())
    exp_colors = ["#2980B9", "#E67E22", "#27AE60"]

    for metric, ylabel, title_sfx in [
        ("R2",   "R² Score",  "R² Score — All Experiments (↑ higher = better)"),
        ("RMSE", "RMSE",      "RMSE — All Experiments (↓ lower = better)"),
        ("MAE",  "MAE",       "MAE — All Experiments (↓ lower = better)"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        x     = np.arange(len(model_names_all))
        width = 0.22

        for i, ((ekey, einfo), ecolor) in enumerate(zip(exp_list, exp_colors)):
            vals = []
            for mname in model_names_all:
                subset = combined_df[
                    (combined_df["Experiment"] == einfo["label"]) &
                    (combined_df["Model"] == mname)
                ]
                vals.append(float(subset[metric].values[0]) if len(subset) else 0)

            bars = ax.bar(x + i * width, vals, width,
                          label=einfo["short"], color=ecolor, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.005,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7, color="#333")

        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names_all, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title_sfx, fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=10, title="Experiment")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"all_{metric.lower()}_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ all_{metric.lower()}_comparison.png saved")

    # ── Print final summary table ──────────────────────────────
    print("\n" + "=" * 62)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 62)
    summary = combined_df[["Experiment", "Model", "R2", "RMSE", "MAE"]]
    print(summary.to_string(index=False))

# ─── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  ALL EVALUATIONS COMPLETE!")
print("=" * 62)
print("""
  results/
  ├── exp1_results.csv
  ├── exp2_results.csv
  ├── exp3_results.csv
  ├── all_experiments_comparison.csv
  └── plots/
      ├── exp1_metrics_comparison.png
      ├── exp2_metrics_comparison.png
      ├── exp3_metrics_comparison.png
      ├── all_r2_comparison.png
      ├── all_rmse_comparison.png
      ├── all_mae_comparison.png
      └── actual_vs_predicted/
          └── (18 plots — 6 models × 3 experiments)
""")

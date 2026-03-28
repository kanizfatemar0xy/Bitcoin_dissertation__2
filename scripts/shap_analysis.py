"""
Script 4: shap_analysis.py — Final Fixed Version
==================================================
LSTM error fix: TensorListStack gradient issue
→ LSTM + NN dono ke liye sirf KernelExplainer use karo

- RF, XGB, GB  → TreeExplainer  (fast)
- SVR          → KernelExplainer
- NN           → KernelExplainer (model.predict wrapper)
- LSTM         → KernelExplainer (sequence wrapper, avg over timesteps)

Run: py shap_analysis.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap, pickle, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model

BASE_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
SHAP_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shap")

EXPERIMENTS = {
    "exp1": {"file": os.path.join(BASE_DIR,"master_price_news.csv"),   "label":"Exp 1 — Price + News"},
    "exp2": {"file": os.path.join(BASE_DIR,"master_price_tweets.csv"), "label":"Exp 2 — Price + Tweets"},
    "exp3": {"file": os.path.join(BASE_DIR,"master_dataset.csv"),      "label":"Exp 3 — All"},
}
TARGET     = "Volatility_7d"
TEST_SPLIT = 0.20
TOP_N      = 15
BG         = "#FAFAFA"
TIMESTEPS  = 10

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_pkl(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:  return pickle.load(f)

def make_sequences(X, ts=10):
    return np.array([X[i:i+ts] for i in range(len(X) - ts)])

def plot_bar(sv, feat_names, title, save_path, color="#4C72B0"):
    sv  = np.array(sv)
    ma  = np.abs(sv).mean(axis=0)
    idx = np.argsort(ma)[::-1][:TOP_N]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feat_names[i] for i in idx][::-1], ma[idx][::-1], color=color, alpha=0.85)
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ {os.path.basename(save_path)}")
    return ma

def plot_beeswarm(model, X_sc, feat_names, title, save_path):
    try:
        expl = shap.Explainer(model, X_sc, feature_names=feat_names)
        sv   = expl(X_sc)
        ma   = np.abs(sv.values).mean(axis=0)
        idx  = np.argsort(ma)[::-1][:TOP_N]
        obj  = shap.Explanation(
            values       = sv.values[:, idx],
            base_values  = sv.base_values,
            data         = sv.data[:, idx],
            feature_names= [feat_names[i] for i in idx]
        )
        fig, _ = plt.subplots(figsize=(9, 6))
        shap.plots.beeswarm(obj, max_display=TOP_N, show=False)
        plt.title(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close("all")
        print(f"   ✅ {os.path.basename(save_path)}")
    except Exception as e:
        print(f"   ⚠  Beeswarm: {e}")

# ─── KernelExplainer wrappers ─────────────────────────────────────────────────
def shap_nn_kernel(model, X_sc, n_bg=50, n_samp=100):
    """NN: simple predict wrapper"""
    bg   = X_sc[:min(n_bg,   len(X_sc))].astype(np.float32)
    samp = X_sc[:min(n_samp, len(X_sc))].astype(np.float32)

    def predict_fn(x):
        return model.predict(x.astype(np.float32), verbose=0).ravel()

    expl = shap.KernelExplainer(predict_fn, bg)
    sv   = expl.shap_values(samp, nsamples=100)
    return np.array(sv)

def shap_lstm_kernel(model, X_sc, n_bg=20, n_samp=50):
    """
    LSTM: flatten (n, TS, feat) → (n, TS*feat) for KernelExplainer,
    then reshape predictions back.
    Average SHAP over timesteps → (n, feat)
    """
    X_seq = make_sequences(X_sc, TIMESTEPS)          # (n, TS, feat)
    n_feat= X_seq.shape[2]

    bg_seq   = X_seq[:min(n_bg,   len(X_seq))].astype(np.float32)
    samp_seq = X_seq[:min(n_samp, len(X_seq))].astype(np.float32)

    # Flatten for KernelExplainer
    bg_flat   = bg_seq.reshape(len(bg_seq),   -1)
    samp_flat = samp_seq.reshape(len(samp_seq), -1)

    def predict_fn(x_flat):
        x3d = x_flat.reshape(-1, TIMESTEPS, n_feat).astype(np.float32)
        return model.predict(x3d, verbose=0).ravel()

    print("   [KernelExplainer running — this takes ~5 min per experiment...]")
    expl    = shap.KernelExplainer(predict_fn, bg_flat)
    sv_flat = expl.shap_values(samp_flat, nsamples=100)  # (n, TS*feat)

    # Average over timesteps
    sv_3d = np.array(sv_flat).reshape(-1, TIMESTEPS, n_feat)
    sv_2d = sv_3d.mean(axis=1)                            # (n, feat)
    return sv_2d

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  SHAP ANALYSIS — ALL EXPERIMENTS (Final Fixed)")
print("=" * 60)

for exp_key, exp_info in EXPERIMENTS.items():
    print(f"\n{'─'*60}\n  {exp_info['label']}\n{'─'*60}")

    if not os.path.exists(exp_info["file"]):
        print("  ⚠  Dataset not found"); continue

    df        = pd.read_csv(exp_info["file"], parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ["Date", TARGET]]
    X         = df[feat_cols].values
    split     = int(len(X) * (1 - TEST_SPLIT))
    X_test    = X[split:]

    mdl_dir   = os.path.join(MODELS_DIR, exp_key)
    scaler_X  = load_pkl(os.path.join(mdl_dir, "scaler.pkl"))
    if scaler_X is None:
        print("  ⚠  Scaler not found — run train_all_experiments.py first!"); continue

    X_test_sc = scaler_X.transform(X_test)
    out_dir   = os.path.join(SHAP_DIR, exp_key)
    os.makedirs(out_dir, exist_ok=True)
    all_imp   = {}

    # ── 1. Random Forest ──────────────────────────────────────
    print(f"\n  [1/6] Random Forest — TreeExplainer")
    m = load_pkl(os.path.join(mdl_dir, "random_forest_model.pkl"))
    if m:
        try:
            sv = shap.TreeExplainer(m).shap_values(X_test_sc)
            all_imp["Random Forest"] = plot_bar(sv, feat_cols,
                f"SHAP — Random Forest ({exp_info['label']})",
                os.path.join(out_dir, "shap_random_forest_bar.png"), "#E67E22")
            plot_beeswarm(m, X_test_sc, feat_cols,
                f"SHAP Beeswarm — Random Forest",
                os.path.join(out_dir, "shap_random_forest_beeswarm.png"))
        except Exception as e: print(f"   ⚠  {e}")
    else: print("   ⚠  Not found")

    # ── 2. XGBoost ────────────────────────────────────────────
    print(f"\n  [2/6] XGBoost — TreeExplainer")
    m = load_pkl(os.path.join(mdl_dir, "xgboost_model.pkl"))
    if m:
        try:
            sv = shap.TreeExplainer(m).shap_values(X_test_sc)
            all_imp["XGBoost"] = plot_bar(sv, feat_cols,
                f"SHAP — XGBoost ({exp_info['label']})",
                os.path.join(out_dir, "shap_xgboost_bar.png"), "#2980B9")
            plot_beeswarm(m, X_test_sc, feat_cols,
                f"SHAP Beeswarm — XGBoost",
                os.path.join(out_dir, "shap_xgboost_beeswarm.png"))
        except Exception as e: print(f"   ⚠  {e}")
    else: print("   ⚠  Not found")

    # ── 3. Gradient Boosting ──────────────────────────────────
    print(f"\n  [3/6] Gradient Boosting — TreeExplainer")
    m = load_pkl(os.path.join(mdl_dir, "gradient_boosting_model.pkl"))
    if m:
        try:
            sv = shap.TreeExplainer(m).shap_values(X_test_sc)
            all_imp["Gradient Boosting"] = plot_bar(sv, feat_cols,
                f"SHAP — Gradient Boosting ({exp_info['label']})",
                os.path.join(out_dir, "shap_gradient_boosting_bar.png"), "#27AE60")
            plot_beeswarm(m, X_test_sc, feat_cols,
                f"SHAP Beeswarm — Gradient Boosting",
                os.path.join(out_dir, "shap_gradient_boosting_beeswarm.png"))
        except Exception as e: print(f"   ⚠  {e}")
    else: print("   ⚠  Not found")

    # ── 4. SVR ────────────────────────────────────────────────
    print(f"\n  [4/6] SVR — KernelExplainer")
    m = load_pkl(os.path.join(mdl_dir, "svr_model.pkl"))
    if m:
        try:
            bg   = shap.sample(X_test_sc, min(50, len(X_test_sc)), random_state=42)
            samp = X_test_sc[:min(100, len(X_test_sc))]
            expl = shap.KernelExplainer(m.predict, bg)
            sv   = expl.shap_values(samp, nsamples=100)
            all_imp["SVR"] = plot_bar(sv, feat_cols,
                f"SHAP — SVR ({exp_info['label']})",
                os.path.join(out_dir, "shap_svr_bar.png"), "#8E44AD")
        except Exception as e: print(f"   ⚠  {e}")
    else: print("   ⚠  Not found")

    # ── 5. Neural Network — KernelExplainer ──────────────────
    print(f"\n  [5/6] Neural Network — KernelExplainer")
    nn_path = os.path.join(mdl_dir, "nn_model.keras")
    if os.path.exists(nn_path):
        try:
            nn = load_model(nn_path)
            sv = shap_nn_kernel(nn, X_test_sc)
            all_imp["Neural Network"] = plot_bar(sv, feat_cols,
                f"SHAP — Neural Network ({exp_info['label']})",
                os.path.join(out_dir, "shap_nn_bar.png"), "#16A085")
        except Exception as e: print(f"   ⚠  NN error: {e}")
    else: print("   ⚠  nn_model.keras not found")

    # ── 6. LSTM — KernelExplainer (sequence) ─────────────────
    print(f"\n  [6/6] LSTM — KernelExplainer (sequence)")
    lstm_path = os.path.join(mdl_dir, "lstm_model.keras")
    if os.path.exists(lstm_path):
        try:
            lstm = load_model(lstm_path)
            sv2d = shap_lstm_kernel(lstm, X_test_sc)
            all_imp["LSTM"] = plot_bar(sv2d, feat_cols,
                f"SHAP — LSTM ({exp_info['label']})",
                os.path.join(out_dir, "shap_lstm_bar.png"), "#E74C3C")
        except Exception as e: print(f"   ⚠  LSTM error: {e}")
    else: print("   ⚠  lstm_model.keras not found")

    # ── All Models Comparison ─────────────────────────────────
    if len(all_imp) >= 2:
        print(f"\n  [Comparison Plot]")
        try:
            imp_df  = pd.DataFrame(all_imp, index=feat_cols)
            imp_df["avg"] = imp_df.mean(axis=1)
            top_f   = imp_df.nlargest(TOP_N, "avg").index.tolist()
            plot_df = imp_df.loc[top_f].drop(columns="avg")
            colors  = ["#E67E22","#2980B9","#27AE60","#8E44AD","#16A085","#E74C3C"]
            x, w    = np.arange(len(top_f)), 0.13
            fig, ax = plt.subplots(figsize=(14, 6))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            for i, (col, clr) in enumerate(zip(plot_df.columns, colors)):
                ax.bar(x + i*w, plot_df[col], w, label=col, color=clr, alpha=0.85)
            ax.set_xticks(x + w*(len(plot_df.columns)-1)/2)
            ax.set_xticklabels(top_f, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Mean |SHAP Value|", fontsize=10)
            ax.set_title(f"SHAP All Models — {exp_info['label']}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "shap_all_comparison.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)
            print("   ✅ shap_all_comparison.png")

            imp_df.sort_values("avg", ascending=False).to_csv(
                os.path.join(out_dir, "shap_feature_importance.csv"))
            print("   ✅ shap_feature_importance.csv")
        except Exception as e:
            print(f"   ⚠  Comparison: {e}")

    print(f"\n  ✅ {exp_key} complete!")

print("\n" + "="*60)
print("  ALL SHAP DONE!")
print("="*60)
print("""
  shap/
  ├── exp1/  shap_random_forest_bar/beeswarm
  │          shap_xgboost_bar/beeswarm
  │          shap_gradient_boosting_bar/beeswarm
  │          shap_svr_bar
  │          shap_nn_bar         ← NEW
  │          shap_lstm_bar        ← NEW
  │          shap_all_comparison
  │          shap_feature_importance.csv
  ├── exp2/  (same)
  └── exp3/  (same)
""")
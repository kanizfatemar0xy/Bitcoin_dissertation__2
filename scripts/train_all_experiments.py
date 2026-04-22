"""
Script 3: train_all_experiments.py
=====================================
Three experiments are trained in 6 models:
  - Random Forest
  - XGBoost
  - Gradient Boosting
  - SVR
  - LSTM
  - Neural Network

Output:
  models/
  ├── exp1/  (Price + News)
  ├── exp2/  (Price + Tweets)
  └── exp3/  (Price + Tweets + News)

Run:  py train_all_experiments.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm             import SVR
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
from xgboost                 import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

EXPERIMENTS = {
    "exp1": {
        "file"  : os.path.join(BASE_DIR, "master_price_news.csv"),
        "label" : "Experiment 1 — Price + News",
    },
    "exp2": {
        "file"  : os.path.join(BASE_DIR, "master_price_tweets.csv"),
        "label" : "Experiment 2 — Price + Tweets",
    },
    "exp3": {
        "file"  : os.path.join(BASE_DIR, "master_dataset.csv"),
        "label" : "Experiment 3 — Price + Tweets + News",
    },
}

TARGET = "Volatility_7d"
SEED   = 42
TEST_SPLIT = 0.20

# ─── LSTM sequence builder ────────────────────────────────────────────────────
def make_sequences(X, y, timesteps=10):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i : i + timesteps])
        ys.append(y[i + timesteps])
    return np.array(Xs), np.array(ys)

# ─── Metrics helper ───────────────────────────────────────────────────────────
def metrics(y_true, y_pred, name):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"      {name:<25}  R²={r2:+.4f}  RMSE={rmse:.6f}  MAE={mae:.6f}")
    return {"model": name, "R2": round(r2,4), "RMSE": round(rmse,6), "MAE": round(mae,6)}

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TRAIN ALL EXPERIMENTS")
print("=" * 60)

all_results = []

for exp_key, exp_info in EXPERIMENTS.items():

    print(f"\n{'─'*60}")
    print(f"  {exp_info['label']}")
    print(f"{'─'*60}")

    # ── Load dataset ──────────────────────────────────────────
    if not os.path.exists(exp_info["file"]):
        print(f"   File not found: {exp_info['file']}")
        print(f"     Run_master_datasets.py")
        continue

    df = pd.read_csv(exp_info["file"], parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Drop non-feature columns
    drop_cols = ["Date", TARGET]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[TARGET].values

    # ── Train/Test split (chronological) ──────────────────────
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Rows     : {len(df):,}  |  Features: {len(feature_cols)}")
    print(f"  Train    : {len(X_train):,}  |  Test    : {len(X_test):,}")
    print(f"  Date     : {df['Date'].min().date()} → {df['Date'].max().date()}")

    # ── Scale ─────────────────────────────────────────────────
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
    y_test_sc  = scaler_y.transform(y_test.reshape(-1,1)).ravel()

    # ── Output folder ─────────────────────────────────────────
    out_dir = os.path.join(MODELS_DIR, exp_key)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Training models → models/{exp_key}/\n")
    exp_results = []

    # ── 1. Random Forest ──────────────────────────────────────
    print("  [1/6] Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_sc, y_train_sc)
    pred = scaler_y.inverse_transform(rf.predict(X_test_sc).reshape(-1,1)).ravel()
    exp_results.append(metrics(y_test, pred, "Random Forest"))
    with open(os.path.join(out_dir, "random_forest_model.pkl"), "wb") as f:
        pickle.dump(rf, f)

    # ── 2. XGBoost ────────────────────────────────────────────
    print("  [2/6] XGBoost...")
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05,
                       max_depth=6, random_state=SEED,
                       verbosity=0, n_jobs=-1)
    xgb.fit(X_train_sc, y_train_sc)
    pred = scaler_y.inverse_transform(xgb.predict(X_test_sc).reshape(-1,1)).ravel()
    exp_results.append(metrics(y_test, pred, "XGBoost"))
    with open(os.path.join(out_dir, "xgboost_model.pkl"), "wb") as f:
        pickle.dump(xgb, f)

    # ── 3. Gradient Boosting ──────────────────────────────────
    print("  [3/6] Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=4, random_state=SEED)
    gb.fit(X_train_sc, y_train_sc)
    pred = scaler_y.inverse_transform(gb.predict(X_test_sc).reshape(-1,1)).ravel()
    exp_results.append(metrics(y_test, pred, "Gradient Boosting"))
    with open(os.path.join(out_dir, "gradient_boosting_model.pkl"), "wb") as f:
        pickle.dump(gb, f)

    # ── 4. SVR ────────────────────────────────────────────────
    print("  [4/6] SVR...")
    svr = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.01)
    svr.fit(X_train_sc, y_train_sc)
    pred = scaler_y.inverse_transform(svr.predict(X_test_sc).reshape(-1,1)).ravel()
    exp_results.append(metrics(y_test, pred, "SVR"))
    with open(os.path.join(out_dir, "svr_model.pkl"), "wb") as f:
        pickle.dump(svr, f)

    # ── Save scalers ─────────────────────────────────────────
    with open(os.path.join(out_dir, "scaler.pkl"),   "wb") as f: pickle.dump(scaler_X, f)
    with open(os.path.join(out_dir, "y_scaler.pkl"), "wb") as f: pickle.dump(scaler_y, f)

    # ── 5. LSTM ───────────────────────────────────────────────
    print("  [5/6] LSTM...")
    TIMESTEPS = 10
    X_seq_tr, y_seq_tr = make_sequences(X_train_sc, y_train_sc, TIMESTEPS)
    X_seq_te, y_seq_te = make_sequences(X_test_sc,  y_test_sc,  TIMESTEPS)

    lstm_model = Sequential([
        Input(shape=(TIMESTEPS, X_train_sc.shape[1])),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    lstm_model.fit(X_seq_tr, y_seq_tr,
                   validation_split=0.1,
                   epochs=100, batch_size=32,
                   callbacks=[es], verbose=0)

    pred_sc = lstm_model.predict(X_seq_te, verbose=0).ravel()
    pred    = scaler_y.inverse_transform(pred_sc.reshape(-1,1)).ravel()
    y_test_lstm = scaler_y.inverse_transform(y_seq_te.reshape(-1,1)).ravel()
    exp_results.append(metrics(y_test_lstm, pred, "LSTM"))
    lstm_model.save(os.path.join(out_dir, "lstm_model.keras"))

    # ── 6. Neural Network ─────────────────────────────────────
    print("  [6/6] Neural Network...")
    nn_model = Sequential([
        Input(shape=(X_train_sc.shape[1],)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    nn_model.compile(optimizer="adam", loss="mse")
    es2 = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    nn_model.fit(X_train_sc, y_train_sc,
                 validation_split=0.1,
                 epochs=100, batch_size=32,
                 callbacks=[es2], verbose=0)

    pred_sc = nn_model.predict(X_test_sc, verbose=0).ravel()
    pred    = scaler_y.inverse_transform(pred_sc.reshape(-1,1)).ravel()
    exp_results.append(metrics(y_test, pred, "Neural Network"))
    nn_model.save(os.path.join(out_dir, "nn_model.keras"))

    # ── Save results CSV ──────────────────────────────────────
    res_df = pd.DataFrame(exp_results)
    res_df["experiment"] = exp_key
    res_df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    all_results.extend(exp_results)

    best = max(exp_results, key=lambda x: x["R2"])
    print(f"\n  {exp_key} done!  Best model: {best['model']} (R²={best['R2']})")

# ─── Save combined results ────────────────────────────────────────────────────
combined_path = os.path.join(MODELS_DIR, "all_experiments_results.csv")
os.makedirs(MODELS_DIR, exist_ok=True)
pd.DataFrame(all_results).to_csv(combined_path, index=False)

# ─── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL EXPERIMENTS COMPLETE!")
print("=" * 60)
print("""
  models/
  ├── exp1/
  │   ├── random_forest_model.pkl
  │   ├── xgboost_model.pkl
  │   ├── gradient_boosting_model.pkl
  │   ├── svr_model.pkl
  │   ├── lstm_model.keras
  │   ├── nn_model.keras
  │   ├── scaler.pkl
  │   ├── y_scaler.pkl
  │   └── results.csv
  ├── exp2/   (same files)
  ├── exp3/   (same files)
  └── all_experiments_results.csv
""")
print(" Models saved!\n")

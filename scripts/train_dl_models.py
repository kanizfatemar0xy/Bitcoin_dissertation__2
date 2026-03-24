import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

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
# STEP 1: Load Data
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Master Dataset")
print("=" * 60)

df = pd.read_csv(MASTER_FILE, parse_dates=["Date"])
print(f"  Shape      : {df.shape}")
print(f"  Date range : {df['Date'].min().date()} --> {df['Date'].max().date()}")

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

# ══════════════════════════════════════════════════════════════
# STEP 2: Scale
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Scaling (Loading saved scaler)")
print("=" * 60)

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
X_scaled = scaler.transform(X)

# Scale target too for LSTM
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
joblib.dump(y_scaler, os.path.join(MODELS_DIR, "y_scaler.pkl"))
print("  ✅ Scaler loaded, y_scaler saved.")

# ══════════════════════════════════════════════════════════════
# STEP 3: Train/Test Split
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Train/Test Split (80/20)")
print("=" * 60)

split = int(len(X_scaled) * 0.80)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]
y_test_orig     = y[split:]
dates_test      = dates[split:]

print(f"  Train : {len(X_train)} rows")
print(f"  Test  : {len(X_test)}  rows")

# ══════════════════════════════════════════════════════════════
# STEP 4: Reshape for LSTM [samples, timesteps, features]
# ══════════════════════════════════════════════════════════════
SEQ_LEN = 7   # 7 days lookback

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  SEQ_LEN)
y_test_orig_seq          = y_test_orig[SEQ_LEN:]
dates_test_seq           = dates_test[SEQ_LEN:]

print(f"\n  LSTM sequence shape : {X_train_seq.shape}")

# ── Helper ────────────────────────────────────────────────────
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

def save_pred_plot(name, dates, y_true, y_pred, filename):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, y_true, color="black", linewidth=1.2, label="Actual Volatility")
    ax.plot(dates, y_pred, color="gray",  linewidth=1.2,
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

def save_loss_plot(history, name, filename):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"],     color="black", linewidth=1.2, label="Train Loss")
    ax.plot(history.history["val_loss"], color="gray",  linewidth=1.2,
            linestyle="--", label="Val Loss")
    ax.set_title(f"{name} — Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Loss plot saved: plots/ml_results/{filename}")

results = []

# ══════════════════════════════════════════════════════════════
# STEP 5: LSTM Model
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: LSTM Model")
print("=" * 60)

n_features = X_train_seq.shape[2]

def build_lstm(units1=64, units2=32, dropout=0.2, lr=0.001):
    model = Sequential([
        LSTM(units1, return_sequences=True,
             input_shape=(SEQ_LEN, n_features)),
        Dropout(dropout),
        BatchNormalization(),
        LSTM(units2, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
    return model

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=7, min_lr=1e-6, verbose=0)
]

print("  Training LSTM (base)...")
lstm_model = build_lstm()
history_lstm = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=100, batch_size=16,
    validation_split=0.1,
    callbacks=callbacks, verbose=0
)
print(f"  Stopped at epoch: {len(history_lstm.history['loss'])}")

lstm_pred_scaled = lstm_model.predict(X_test_seq, verbose=0).flatten()
lstm_pred = y_scaler.inverse_transform(
    lstm_pred_scaled.reshape(-1,1)).flatten()
res_lstm = evaluate("LSTM", y_test_orig_seq, lstm_pred)
results.append(res_lstm)
save_pred_plot("LSTM", dates_test_seq, y_test_orig_seq,
               lstm_pred, "lstm_actual_vs_pred.png")
save_loss_plot(history_lstm, "LSTM", "lstm_loss.png")

# Fine-tune LSTM
print("\n  Fine-tuning LSTM (larger)...")
lstm_tuned = build_lstm(units1=128, units2=64, dropout=0.3, lr=0.0005)
history_lstm_tuned = lstm_tuned.fit(
    X_train_seq, y_train_seq,
    epochs=150, batch_size=16,
    validation_split=0.1,
    callbacks=callbacks, verbose=0
)
print(f"  Stopped at epoch: {len(history_lstm_tuned.history['loss'])}")

lstm_tuned_pred_sc = lstm_tuned.predict(X_test_seq, verbose=0).flatten()
lstm_tuned_pred    = y_scaler.inverse_transform(
    lstm_tuned_pred_sc.reshape(-1,1)).flatten()
res_lstm_tuned = evaluate("LSTM (Tuned)", y_test_orig_seq, lstm_tuned_pred)
results.append(res_lstm_tuned)
save_pred_plot("LSTM (Tuned)", dates_test_seq, y_test_orig_seq,
               lstm_tuned_pred, "lstm_tuned_actual_vs_pred.png")
save_loss_plot(history_lstm_tuned, "LSTM (Tuned)", "lstm_tuned_loss.png")

lstm_tuned.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
print("  ✅ Saved: models/lstm_model.keras")

# ══════════════════════════════════════════════════════════════
# STEP 6: Neural Network (NN)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Neural Network (NN)")
print("=" * 60)

# NN uses flat input (not sequences)
X_train_flat = X_train
X_test_flat  = X_test
y_test_orig_flat = y_test_orig

def build_nn(units=[128, 64, 32], dropout=0.2, lr=0.001):
    model = Sequential()
    model.add(Dense(units[0], activation="relu",
                    input_shape=(X_train_flat.shape[1],)))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    for u in units[1:]:
        model.add(Dense(u, activation="relu"))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
    return model

callbacks_nn = [
    EarlyStopping(monitor="val_loss", patience=15,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=7, min_lr=1e-6, verbose=0)
]

print("  Training NN (base)...")
nn_model = build_nn()
history_nn = nn_model.fit(
    X_train_flat, y_train,
    epochs=100, batch_size=16,
    validation_split=0.1,
    callbacks=callbacks_nn, verbose=0
)
print(f"  Stopped at epoch: {len(history_nn.history['loss'])}")

nn_pred_scaled = nn_model.predict(X_test_flat, verbose=0).flatten()
nn_pred = y_scaler.inverse_transform(
    nn_pred_scaled.reshape(-1,1)).flatten()
res_nn = evaluate("Neural Network", y_test_orig_flat, nn_pred)
results.append(res_nn)
save_pred_plot("Neural Network", dates_test, y_test_orig_flat,
               nn_pred, "nn_actual_vs_pred.png")
save_loss_plot(history_nn, "Neural Network", "nn_loss.png")

# Fine-tune NN
print("\n  Fine-tuning NN (deeper)...")
nn_tuned = build_nn(units=[256, 128, 64, 32], dropout=0.3, lr=0.0005)
history_nn_tuned = nn_tuned.fit(
    X_train_flat, y_train,
    epochs=150, batch_size=16,
    validation_split=0.1,
    callbacks=callbacks_nn, verbose=0
)
print(f"  Stopped at epoch: {len(history_nn_tuned.history['loss'])}")

nn_tuned_pred_sc = nn_tuned.predict(X_test_flat, verbose=0).flatten()
nn_tuned_pred    = y_scaler.inverse_transform(
    nn_tuned_pred_sc.reshape(-1,1)).flatten()
res_nn_tuned = evaluate("Neural Network (Tuned)", y_test_orig_flat, nn_tuned_pred)
results.append(res_nn_tuned)
save_pred_plot("Neural Network (Tuned)", dates_test, y_test_orig_flat,
               nn_tuned_pred, "nn_tuned_actual_vs_pred.png")
save_loss_plot(history_nn_tuned, "Neural Network (Tuned)", "nn_tuned_loss.png")

nn_tuned.save(os.path.join(MODELS_DIR, "nn_model.keras"))
print("  ✅ Saved: models/nn_model.keras")

# ══════════════════════════════════════════════════════════════
# STEP 7: Final Comparison — ALL 7 Models
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Final Comparison — All 7 Models")
print("=" * 60)

# Load ML results
ml_results = pd.read_csv(os.path.join(DATA, "ml_results.csv"))
ml_tuned   = ml_results[ml_results["Model"].str.contains("Tuned")]

# DL results
dl_results = pd.DataFrame(results)
dl_tuned   = dl_results[dl_results["Model"].str.contains("Tuned")]

# Combine
all_tuned = pd.concat([ml_tuned, dl_tuned], ignore_index=True)
all_tuned["Model"] = all_tuned["Model"].str.replace(" (Tuned)", "", regex=False)
all_tuned = all_tuned.sort_values("RMSE").reset_index(drop=True)
all_tuned.to_csv(os.path.join(DATA, "all_models_results.csv"), index=False)

print("\n  📊 ALL 7 MODELS — Final Ranking:")
print(all_tuned.to_string(index=False))

# Final comparison plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("All Models Comparison (Tuned)", fontsize=13, fontweight="bold")

metrics = ["RMSE", "MAE", "R2"]
titles  = ["RMSE ↓ lower is better",
           "MAE  ↓ lower is better",
           "R²   ↑ higher is better"]

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i]
    bars = ax.bar(all_tuned["Model"], all_tuned[metric],
                  color="gray", edgecolor="black", width=0.5)
    # Highlight best
    best_idx = all_tuned[metric].idxmin() if metric != "R2" \
               else all_tuned[metric].idxmax()
    bars[best_idx].set_color("black")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticklabels(all_tuned["Model"], rotation=30, ha="right")
    for bar, val in zip(bars, all_tuned[metric]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "all_models_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✅ Saved: plots/ml_results/all_models_comparison.png")

# ── Final Summary ─────────────────────────────────────────────
best_model = all_tuned.iloc[0]["Model"]
print("\n" + "=" * 60)
print("FINAL SUMMARY — ALL 7 MODELS")
print("=" * 60)
print(f"""
  ML Models  : Random Forest, XGBoost, Gradient Boosting, SVR
  DL Models  : LSTM, Neural Network
  Scaler     : MinMaxScaler
  Fine-tuning: GridSearchCV (ML) + EarlyStopping (DL)
  Metrics    : RMSE, MAE, R²

  🏆 Best Overall Model : {best_model}

  Saved Models:
    models/scaler.pkl
    models/random_forest_model.pkl
    models/xgboost_model.pkl
    models/gradient_boosting_model.pkl
    models/svr_model.pkl
    models/lstm_model.keras
    models/nn_model.keras

  Saved Results:
    data/ml_results.csv
    data/all_models_results.csv

  Next Step → SHAP (Explainable AI)
""")

"""
app.py — Bitcoin Volatility Predictor (Fixed)
Fix 1: Accurate prediction — proper feature alignment
Fix 2: LSTM correct 3D input handling
Fix 3: Any-date prediction via nearest neighbor
Fix 4: RMSE/MAE on dashboard
"""
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import yfinance as yf
import urllib.request
import xml.etree.ElementTree as ET
from textblob import TextBlob
import os, pickle, json, warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)

DATA_DIR    = "data"
MODELS_DIR  = "models"
SHAP_DIR    = "shap"
RAG_DIR     = "rag"
RESULTS_DIR = "results"

EXPERIMENT_FILES = {
    "exp1": os.path.join(DATA_DIR, "master_price_news.csv"),
    "exp2": os.path.join(DATA_DIR, "master_price_tweets.csv"),
    "exp3": os.path.join(DATA_DIR, "master_dataset.csv"),
}
EXPERIMENT_LABELS = {
    "exp1": "Price + News",
    "exp2": "Price + Tweets",
    "exp3": "Price + Tweets + News",
}
MODEL_DISPLAY = {
    "xgboost"          : "XGBoost",
    "random_forest"    : "Random Forest",
    "gradient_boosting": "Gradient Boosting (Lowest Error)",
    "svr"              : "SVR",
    "lstm"             : "LSTM (Deep Learning)",
    "nn"               : "Neural Network",
}
MODEL_RESULTS_NAME = {
    "xgboost"          : "XGBoost",
    "random_forest"    : "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svr"              : "SVR",
    "lstm"             : "LSTM",
    "nn"               : "Neural Network",
}
TARGET    = "Volatility_7d"
TIMESTEPS = 10   # LSTM sequence length — must match training

# ─── Cache dataframes to avoid re-reading CSV ─────────────────────────────────
_df_cache = {}
def get_df(exp_key):
    if exp_key not in _df_cache:
        p = EXPERIMENT_FILES[exp_key]
        df = pd.read_csv(p, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        _df_cache[exp_key] = df
    return _df_cache[exp_key].copy()

def get_feat_cols(exp_key):
    df = get_df(exp_key)
    return [c for c in df.columns if c not in ["Date", TARGET]]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_pkl(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:  return pickle.load(f)

def get_model_metrics(exp_key, model_choice):
    p = os.path.join(RESULTS_DIR, f"{exp_key}_results.csv")
    if not os.path.exists(p): return "N/A","N/A","N/A"
    try:
        df  = pd.read_csv(p)
        mn  = MODEL_RESULTS_NAME.get(model_choice, model_choice)
        row = df[df["Model"]==mn]
        if row.empty: return "N/A","N/A","N/A"
        r = row.iloc[0]
        return str(round(float(r["R2"]),4)), str(round(float(r["RMSE"]),6)), str(round(float(r["MAE"]),6))
    except: return "N/A","N/A","N/A"

# ─── Feature row for any date ─────────────────────────────────────────────────
def get_row_for_date(exp_key, target_date_str):
    """
    Returns (feature_dict, used_date_str, is_exact, close_price, volatility,
             news_sent, tweet_sent)
    For LSTM: also returns X_history (last TIMESTEPS rows scaled)
    """
    df       = get_df(exp_key)
    feat_cols= get_feat_cols(exp_key)

    # Try exact match first
    df["Date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
    exact = df[df["Date_str"] == target_date_str]

    if not exact.empty:
        row       = exact.iloc[0]
        is_exact  = True
        used_date = target_date_str
        row_idx   = exact.index[0]
    else:
        # Nearest date
        target_dt        = pd.to_datetime(target_date_str)
        df["_diff"]      = (df["Date"] - target_dt).abs()
        nearest_idx      = df["_diff"].idxmin()
        row              = df.loc[nearest_idx]
        is_exact         = False
        used_date        = row["Date_str"]
        row_idx          = nearest_idx

    # Build feature dict — exactly matching training column order
    fdict = {}
    for c in feat_cols:
        val = row.get(c, 0.0)
        try:    fdict[c] = float(val) if not pd.isna(val) else 0.0
        except: fdict[c] = 0.0

    price      = float(row.get("Close", 0))
    volatility = float(row.get(TARGET, 0))
    news_sent  = float(row.get("News_Avg_Sentiment",  0)) if "News_Avg_Sentiment"  in row.index else None
    tweet_sent = float(row.get("Tweet_Avg_Sentiment", 0)) if "Tweet_Avg_Sentiment" in row.index else None

    # LSTM history: get last TIMESTEPS rows up to and including row_idx
    history_start = max(0, row_idx - TIMESTEPS + 1)
    history_rows  = df.loc[history_start:row_idx, feat_cols].values  # shape (<=TIMESTEPS, n_feat)

    return fdict, used_date, is_exact, price, volatility, news_sent, tweet_sent, history_rows

# ─── Prediction ───────────────────────────────────────────────────────────────
def predict_volatility(exp_key, model_choice, fdict, history_rows=None):
    """
    Accurate prediction:
    - Tree/SVR models: single scaled row
    - LSTM: sequence of last TIMESTEPS rows (3D input)
    - NN: single scaled row
    """
    try:
        mdl_dir  = os.path.join(MODELS_DIR, exp_key)
        scaler_X = load_pkl(os.path.join(mdl_dir, "scaler.pkl"))
        scaler_y = load_pkl(os.path.join(mdl_dir, "y_scaler.pkl"))
        if scaler_X is None:
            return None, "Scaler not found. Run train_all_experiments.py first."

        feat_cols = get_feat_cols(exp_key)
        X_single  = np.array([[fdict.get(c, 0.0) for c in feat_cols]])
        X_sc      = scaler_X.transform(X_single)   # (1, n_feat)

        if model_choice == "lstm":
            model = load_model(os.path.join(mdl_dir, "lstm_model.keras"))

            # Build proper sequence
            if history_rows is not None and len(history_rows) >= 2:
                # Pad if shorter than TIMESTEPS
                seq = history_rows[-TIMESTEPS:]                        # (<=TS, n_feat)
                if len(seq) < TIMESTEPS:
                    pad = np.zeros((TIMESTEPS - len(seq), seq.shape[1]))
                    seq = np.vstack([pad, seq])                        # (TS, n_feat)
                seq_sc = scaler_X.transform(seq)                       # (TS, n_feat)
            else:
                # Fallback: repeat single row
                seq_sc = np.repeat(X_sc, TIMESTEPS, axis=0)           # (TS, n_feat)

            X_lstm = seq_sc.reshape(1, TIMESTEPS, len(feat_cols))      # (1, TS, n_feat)
            p      = float(model.predict(X_lstm, verbose=0)[0][0])

        elif model_choice == "nn":
            model = load_model(os.path.join(mdl_dir, "nn_model.keras"))
            p     = float(model.predict(X_sc, verbose=0)[0][0])

        else:
            model = load_pkl(os.path.join(mdl_dir, f"{model_choice}_model.pkl"))
            p     = float(model.predict(X_sc)[0])

        pred = float(scaler_y.inverse_transform([[p]])[0][0])
        return round(abs(pred), 5), None   # abs: volatility always positive

    except Exception as e:
        return None, str(e)

# ─── Live data ────────────────────────────────────────────────────────────────
def get_live_crypto_data():
    btc   = yf.download("BTC-USD", period="30d", interval="1d", progress=False)
    close = btc["Close"].iloc[:,0] if isinstance(btc.columns, pd.MultiIndex) else btc["Close"]
    close = close.dropna()
    rets  = close.pct_change().dropna()
    vol7  = float(rets.tail(7).std())
    vol30 = float(rets.tail(30).std())
    return (round(float(close.iloc[-1]),2),
            round(vol7, 5),
            round(float(rets.iloc[-1]), 5),
            round(vol30, 5),
            close, rets)

def get_live_news():
    urls = [
        "https://news.google.com/rss/search?q=Bitcoin+when:1d&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=Cryptocurrency+Market+when:1d&hl=en-US&gl=US&ceid=US:en",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ]
    news, total = [], 0.0
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as r:
                root = ET.fromstring(r.read())
            for item in root.findall(".//item")[:3]:
                t = item.find("title")
                if t is not None and t.text and t.text not in news:
                    news.append(t.text)
                    total += TextBlob(t.text).sentiment.polarity
        except: continue
    avg = round(total/len(news), 3) if news else 0.0
    if not news: news = ["Could not fetch live news."]
    return news[:7], avg

def build_live_fdict(exp_key, price, vol7, vol30, daily_return, news_sent):
    """
    Live mode: fill feature dict using live values.
    Uses training column means for lag/rolling features we don't have live.
    """
    feat_cols = get_feat_cols(exp_key)
    df        = get_df(exp_key)

    # Compute column means from training data for missing features
    col_means = df[feat_cols].mean().to_dict()

    fdict = {}
    for c in feat_cols:
        cl = c.lower()
        if   "close"               in cl: fdict[c] = price
        elif "daily_return"        in cl and "lag" not in cl: fdict[c] = daily_return
        elif "volatility_7d"       in cl and "lag" not in cl: fdict[c] = vol7
        elif "volatility_30d"      in cl: fdict[c] = vol30
        elif "news_avg_sentiment"  in cl and "lag" not in cl: fdict[c] = news_sent
        elif "tweet_avg_sentiment" in cl and "lag" not in cl: fdict[c] = 0.0
        elif "open"                in cl: fdict[c] = price * 0.998
        elif "high"                in cl: fdict[c] = price * 1.005
        elif "low"                 in cl: fdict[c] = price * 0.995
        elif "volume"              in cl: fdict[c] = col_means.get(c, 30000000000)
        elif "ma7"                 in cl: fdict[c] = price
        elif "ma30"                in cl: fdict[c] = price
        elif "return_std7"         in cl: fdict[c] = vol7
        elif "price_range"         in cl: fdict[c] = price * 0.01
        else: fdict[c] = col_means.get(c, 0.0)   # use training mean for lag features
    return fdict

# ─── RAG ──────────────────────────────────────────────────────────────────────
def load_rag(exp_key, date_str):
    p = os.path.join(RAG_DIR, exp_key, "rag_narratives.json")
    if not os.path.exists(p): return None, None
    events = json.load(open(p))
    target = datetime.strptime(date_str, "%Y-%m-%d")
    best, bd = None, timedelta(days=9999)
    for ev in events:
        d = datetime.strptime(ev["date"], "%Y-%m-%d")
        if abs(d-target) < bd: best, bd = ev, abs(d-target)
    if best:
        s = (f"On {best['date']}, Bitcoin showed {best['vol_level']} volatility "
             f"(score: {best['volatility']}). Price was ${best['price']:,}. "
             f"Market context: {best['market_context']}.")
        c = "\n".join(best.get("retrieved_docs", []))
        return s, c
    return None, None

def get_shap_img(exp_key, model_choice):
    nm = {"random_forest":"random_forest","xgboost":"xgboost",
          "gradient_boosting":"gradient_boosting","svr":"svr",
          "lstm":"lstm","nn":"nn"}
    mn = nm.get(model_choice)
    if not mn: return None
    for sfx in ["beeswarm","bar"]:
        p = os.path.join(SHAP_DIR, exp_key, f"shap_{mn}_{sfx}.png")
        if os.path.exists(p): return p.replace("\\","/")
    return None

def get_trend(cur, pred):
    if pred is None: return "— Could not compute","text-secondary"
    if pred > cur*1.01: return " Expected to Increase (Higher Risk)","text-danger"
    if pred < cur*0.99: return "Expected to Decrease (Market Stabilizing)","text-primary"
    return " Expected to Remain Stable","text-secondary"

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", models=MODEL_DISPLAY, experiments=EXPERIMENT_LABELS)

@app.route("/api/date_range/<exp_key>")
def date_range(exp_key):
    p = EXPERIMENT_FILES.get(exp_key)
    if not p or not os.path.exists(p):
        return jsonify({"min":"2021-01-01","max":"2024-12-31"})
    df = pd.read_csv(p, parse_dates=["Date"])
    return jsonify({"min": df["Date"].min().strftime("%Y-%m-%d"),
                    "max": df["Date"].max().strftime("%Y-%m-%d")})

@app.route("/shap_image")
def shap_image():
    path = request.args.get("path","")
    if path and os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "Not found", 404

@app.route("/analyze", methods=["POST"])
def analyze():
    target_date  = request.form.get("target_date","")
    model_choice = request.form.get("model_choice","gradient_boosting")
    exp_key      = request.form.get("experiment","exp3")
    model_label  = MODEL_DISPLAY.get(model_choice, model_choice)
    exp_label    = EXPERIMENT_LABELS.get(exp_key, exp_key)
    is_live      = target_date >= (datetime.now()-timedelta(days=1)).strftime("%Y-%m-%d")
    r2, rmse, mae = get_model_metrics(exp_key, model_choice)

    # ── LIVE ──────────────────────────────────────────────────
    if is_live:
        try:
            price, vol7, daily_return, vol30, close_series, ret_series = get_live_crypto_data()
            live_news, news_sent = get_live_news()

            fdict = build_live_fdict(exp_key, price, vol7, vol30, daily_return, news_sent)
            pv, err = predict_volatility(exp_key, model_choice, fdict)
            pvd = pv if pv else "N/A"
            vt, tc = get_trend(vol7, pv)

            rs = (f"On {target_date} (Live), our system fetched real-time market data. "
                  f"The current Bitcoin price is ${price:,}. "
                  f"Using the {model_label} model, the predicted next-day volatility is {pvd}.")
            rc = " LIVE Top News Headlines (Today):\n" + "\n".join(f"- {n}" for n in live_news)

            return render_template("results.html",
                date=f"{target_date} (LIVE DATA)", price=price,
                volatility=vol7, news_sentiment=news_sent, tweet_sentiment="N/A",
                model_used=model_label, experiment=exp_label,
                predicted_vol=pvd, vol_trend=vt, trend_css=tc,
                rag_summary=rs, rag_context=rc,
                shap_img=get_shap_img(exp_key, model_choice),
                is_live=True, r2=r2, rmse=rmse, mae=mae, date_note="")
        except Exception as e:
            return render_template("results.html", error=str(e))

    # ── HISTORICAL (any date) ─────────────────────────────────
    else:
        try:
            cp = EXPERIMENT_FILES.get(exp_key)
            if not cp or not os.path.exists(cp):
                return render_template("results.html",
                    error=f"Dataset for '{exp_label}' not found. Run create_master_datasets.py first.")

            (fdict, used_date, is_exact,
             price, volatility,
             news_sent, tweet_sent,
             history_rows) = get_row_for_date(exp_key, target_date)

            date_note = "" if is_exact else (
                f" {target_date} is not in the dataset — "
                f"prediction based on nearest available date: {used_date}")

            ns = round(news_sent,  3) if news_sent  is not None else "N/A"
            ts = round(tweet_sent, 3) if tweet_sent is not None else "N/A"

            pv, err = predict_volatility(exp_key, model_choice, fdict, history_rows)
            pvd = pv if pv else f"Error: {err}"
            vt, tc  = get_trend(volatility, pv)

            rs, rc = load_rag(exp_key, target_date)
            if not rs:
                rs = (f"On {target_date}, Bitcoin showed volatility of {volatility}. "
                      f"Price: ${price:,}. Predicted next-day: {pvd}.")
                rc = "Run rag_pipeline.py to generate narrative context."

            return render_template("results.html",
                date=f"{target_date} (Historical Data)", price=price,
                volatility=round(volatility,5),
                news_sentiment=ns, tweet_sentiment=ts,
                model_used=model_label, experiment=exp_label,
                predicted_vol=pvd, vol_trend=vt, trend_css=tc,
                rag_summary=rs, rag_context=rc,
                shap_img=get_shap_img(exp_key, model_choice),
                is_live=False, r2=r2, rmse=rmse, mae=mae, date_note=date_note)

        except Exception as e:
            return render_template("results.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
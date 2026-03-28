"""
Script 2: generate_all_plots.py — SMOOTH VERSION
==================================================
Sab sentiment lines 7-day rolling average se smooth hain
Price jaisi clean simple lines!

Run karo:  py generate_all_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PLOTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")

PRICE_FILE  = os.path.join(BASE_DIR, "1_price_data.csv")
TWEETS_FILE = os.path.join(BASE_DIR, "2_tweets_data.csv")
NEWS_FILE   = os.path.join(BASE_DIR, "3_news_data.csv")
EXP1_FILE   = os.path.join(BASE_DIR, "master_price_news.csv")
EXP2_FILE   = os.path.join(BASE_DIR, "master_price_tweets.csv")
EXP3_FILE   = os.path.join(BASE_DIR, "master_dataset.csv")

# ─── Colors ───────────────────────────────────────────────────────────────────
C_PRICE  = "#F4A825"   # Orange
C_TWEETS = "#1DA1F2"   # Blue
C_NEWS   = "#2ECC71"   # Green
BG       = "#FAFAFA"

# ─── Create folders ───────────────────────────────────────────────────────────
for f in [
    os.path.join(PLOTS_DIR, "individual"),
    os.path.join(PLOTS_DIR, "similarity"),
    os.path.join(PLOTS_DIR, "experiments", "exp1_price_news"),
    os.path.join(PLOTS_DIR, "experiments", "exp2_price_tweets"),
    os.path.join(PLOTS_DIR, "experiments", "exp3_all"),
]:
    os.makedirs(f, exist_ok=True)

# ─── Load ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  GENERATE ALL PLOTS  (Smooth Lines)")
print("=" * 55)
print("\nLoading data files...")

price_df  = pd.read_csv(PRICE_FILE,  parse_dates=["Date"])
tweets_df = pd.read_csv(TWEETS_FILE, parse_dates=["Date"])
news_df   = pd.read_csv(NEWS_FILE,   parse_dates=["Date"])

price_df  = price_df[price_df["Date"] >= "2021-01-01"].sort_values("Date").reset_index(drop=True)
tweets_df = tweets_df[tweets_df["Date"] >= "2021-01-01"].sort_values("Date").reset_index(drop=True)
news_df   = news_df[news_df["Date"] >= "2021-01-01"].sort_values("Date").reset_index(drop=True)

# ─── Detect & rename sentiment columns ───────────────────────────────────────
def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

t_col = get_col(tweets_df, ["Tweet_Avg_Sentiment", "Avg_Sentiment"])
n_col = get_col(news_df,   ["News_Avg_Sentiment",  "Avg_Sentiment"])

if t_col != "Tweet_Avg_Sentiment":
    tweets_df = tweets_df.rename(columns={t_col: "Tweet_Avg_Sentiment"})
    t_col = "Tweet_Avg_Sentiment"

if n_col != "News_Avg_Sentiment":
    news_df = news_df.rename(columns={n_col: "News_Avg_Sentiment"})
    n_col = "News_Avg_Sentiment"

# ─── Smooth helper (7-day rolling) ────────────────────────────────────────────
def smooth(series, window=7):
    return series.rolling(window=window, min_periods=1).mean()

# ─── Normalize helper ─────────────────────────────────────────────────────────
def normalize(s):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s * 0
    return (s - mn) / (mx - mn)

# ─── Axis styling ─────────────────────────────────────────────────────────────
def style_ax(ax, title="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="upper left")

def save_fig(fig, folder, name):
    fig.patch.set_facecolor(BG)
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ {name}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — INDIVIDUAL PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Individual Plots (plots/individual/)...")

ind = os.path.join(PLOTS_DIR, "individual")

# Plot 1 — Price
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(price_df["Date"], price_df["Close"],
        color=C_PRICE, linewidth=1.4, label="Close Price (USD)")
style_ax(ax, title="Bitcoin Daily Close Price (2021–2025)", ylabel="Price (USD)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
save_fig(fig, ind, "plot1_price.png")

# Plot 2 — Tweet Sentiment (smoothed)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(tweets_df["Date"], smooth(tweets_df[t_col]),
        color=C_TWEETS, linewidth=1.4, label="Avg Tweet Sentiment (7-day avg)")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
style_ax(ax, title="Bitcoin Twitter Sentiment (2021–2024)", ylabel="Sentiment Score (-1 to +1)")
save_fig(fig, ind, "plot2_tweets.png")

# Plot 3 — News Sentiment (smoothed)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(news_df["Date"], smooth(news_df[n_col]),
        color=C_NEWS, linewidth=1.4, label="Avg News Sentiment (7-day avg)")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
style_ax(ax, title="Bitcoin News Sentiment (2021–2023)", ylabel="Sentiment Score (-1 to +1)")
save_fig(fig, ind, "plot3_news.png")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SIMILARITY PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Similarity Plots (plots/similarity/)...")

sim = os.path.join(PLOTS_DIR, "similarity")

# Price vs News
m = pd.merge(price_df[["Date","Close"]], news_df[["Date", n_col]], on="Date")
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(m["Date"], normalize(m["Close"]),
        color=C_PRICE, linewidth=1.4, label="Price (normalized)")
ax.plot(m["Date"], normalize(smooth(m[n_col])),
        color=C_NEWS, linewidth=1.4, label="News Sentiment (normalized, 7-day avg)")
style_ax(ax, title="Bitcoin Price vs News Sentiment (Normalized)", ylabel="Normalized Value (0–1)")
save_fig(fig, sim, "price_vs_news.png")

# Price vs Tweets
m = pd.merge(price_df[["Date","Close"]], tweets_df[["Date", t_col]], on="Date")
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(m["Date"], normalize(m["Close"]),
        color=C_PRICE, linewidth=1.4, label="Price (normalized)")
ax.plot(m["Date"], normalize(smooth(m[t_col])),
        color=C_TWEETS, linewidth=1.4, label="Tweet Sentiment (normalized, 7-day avg)")
style_ax(ax, title="Bitcoin Price vs Tweet Sentiment (Normalized)", ylabel="Normalized Value (0–1)")
save_fig(fig, sim, "price_vs_tweets.png")

# Price vs Tweets vs News
m = pd.merge(price_df[["Date","Close"]], tweets_df[["Date", t_col]], on="Date")
m = pd.merge(m, news_df[["Date", n_col]], on="Date")
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(m["Date"], normalize(m["Close"]),
        color=C_PRICE, linewidth=1.4, label="Price (normalized)")
ax.plot(m["Date"], normalize(smooth(m[t_col])),
        color=C_TWEETS, linewidth=1.4, label="Tweet Sentiment (normalized, 7-day avg)")
ax.plot(m["Date"], normalize(smooth(m[n_col])),
        color=C_NEWS, linewidth=1.4, label="News Sentiment (normalized, 7-day avg)")
style_ax(ax, title="Bitcoin Price vs Tweet Sentiment vs News Sentiment (Normalized)", ylabel="Normalized Value (0–1)")
save_fig(fig, sim, "price_vs_news_vs_tweets.png")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — EXPERIMENT PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Experiment Plots (plots/experiments/)...")

def plot_experiment(csv_path, exp_dir, sent_cols, sent_labels, sent_colors, exp_title):
    if not os.path.exists(csv_path):
        print(f"   ⚠  Skipped (not found): {os.path.basename(csv_path)}")
        return

    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # Price plot
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["Date"], df["Close"], color=C_PRICE, linewidth=1.4, label="Close Price (USD)")
    style_ax(ax, title=f"{exp_title} — Bitcoin Close Price", ylabel="Price (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    save_fig(fig, exp_dir, "plot_price.png")

    # Sentiment plots (smoothed)
    for col, label, color in zip(sent_cols, sent_labels, sent_colors):
        if col not in df.columns:
            continue
        safe = col.lower().replace("_avg_sentiment","").replace(" ","_")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df["Date"], smooth(df[col]),
                color=color, linewidth=1.4, label=f"{label} (7-day avg)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        style_ax(ax, title=f"{exp_title} — {label}", ylabel="Sentiment Score (-1 to +1)")
        save_fig(fig, exp_dir, f"plot_{safe}_sentiment.png")

    # Similarity plot
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["Date"], normalize(df["Close"]),
            color=C_PRICE, linewidth=1.4, label="Price (normalized)")
    for col, label, color in zip(sent_cols, sent_labels, sent_colors):
        if col in df.columns:
            ax.plot(df["Date"], normalize(smooth(df[col])),
                    color=color, linewidth=1.4, label=f"{label} (normalized, 7-day avg)")
    style_ax(ax, title=f"{exp_title} — Price vs Sentiment (Normalized)", ylabel="Normalized Value (0–1)")
    save_fig(fig, exp_dir, "similarity_plot.png")

# Exp 1
plot_experiment(EXP1_FILE,
    os.path.join(PLOTS_DIR, "experiments", "exp1_price_news"),
    sent_cols=["News_Avg_Sentiment"], sent_labels=["News Sentiment"],
    sent_colors=[C_NEWS], exp_title="Experiment 1 (Price + News)")

# Exp 2
plot_experiment(EXP2_FILE,
    os.path.join(PLOTS_DIR, "experiments", "exp2_price_tweets"),
    sent_cols=["Tweet_Avg_Sentiment"], sent_labels=["Tweet Sentiment"],
    sent_colors=[C_TWEETS], exp_title="Experiment 2 (Price + Tweets)")

# Exp 3
plot_experiment(EXP3_FILE,
    os.path.join(PLOTS_DIR, "experiments", "exp3_all"),
    sent_cols=["Tweet_Avg_Sentiment", "News_Avg_Sentiment"],
    sent_labels=["Tweet Sentiment", "News Sentiment"],
    sent_colors=[C_TWEETS, C_NEWS],
    exp_title="Experiment 3 (Price + Tweets + News)")

# ─── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ALL PLOTS SAVED!")
print("=" * 55)
print("""
  plots/
  ├── individual/
  │   ├── plot1_price.png
  │   ├── plot2_tweets.png
  │   └── plot3_news.png
  ├── similarity/
  │   ├── price_vs_news.png
  │   ├── price_vs_tweets.png
  │   └── price_vs_news_vs_tweets.png
  └── experiments/
      ├── exp1_price_news/   (plot_price + plot_news_sentiment + similarity_plot)
      ├── exp2_price_tweets/ (plot_price + plot_tweets_sentiment + similarity_plot)
      └── exp3_all/          (plot_price + plot_tweet + plot_news + similarity_plot)
""")

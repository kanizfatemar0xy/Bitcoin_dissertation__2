"""
Script 1: create_master_datasets.py
====================================
  script 3 master datasets:
  1. master_price_news.csv     (Experiment 1)
  2. master_price_tweets.csv   (Experiment 2)
  3. master_dataset.csv        (Experiment 3 - All 3)

Run :  py create_master_datasets.py
Output:    Bitcoin/data/ it will be saved in folder
"""

import pandas as pd
import numpy as np
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PRICE_FILE  = os.path.join(BASE_DIR, "1_price_data.csv")
TWEETS_FILE = os.path.join(BASE_DIR, "2_tweets_data.csv")
NEWS_FILE   = os.path.join(BASE_DIR, "3_news_data.csv")

# ─── Helper: Add Lag + Rolling Features ───────────────────────────────────────
def add_features(df, sentiment_cols, lag_days=[1, 2, 3, 7]):
    """
    Price lag features + Rolling features + Sentiment lag features add.
    """
    # Price lag features
    for lag in lag_days:
        df[f"Close_Lag{lag}"]        = df["Close"].shift(lag)
        df[f"Daily_Return_Lag{lag}"] = df["Daily_Return"].shift(lag)

    # Rolling features
    df["MA7"]          = df["Close"].rolling(7).mean()
    df["MA30"]         = df["Close"].rolling(30).mean()
    df["Return_Std7"]  = df["Daily_Return"].rolling(7).std()
    df["Price_Range"]  = df["High"] - df["Low"]
    df["Price_Range_Pct"] = df["Price_Range"] / df["Close"]

    # Sentiment lag features
    for col in sentiment_cols:
        for lag in lag_days:
            df[f"{col}_Lag{lag}"] = df[col].shift(lag)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ─── Load Files ───────────────────────────────────────────────────────────────
print("=" * 55)
print("  MASTER DATASETS CREATOR")
print("=" * 55)

print("\n[1/3] Loading raw files...")

price_df  = pd.read_csv(PRICE_FILE,  parse_dates=["Date"])
tweets_df = pd.read_csv(TWEETS_FILE, parse_dates=["Date"])
news_df   = pd.read_csv(NEWS_FILE,   parse_dates=["Date"])

# Keep only 2021 onwards
price_df  = price_df[price_df["Date"] >= "2021-01-01"].copy()
tweets_df = tweets_df[tweets_df["Date"] >= "2021-01-01"].copy()
news_df   = news_df[news_df["Date"] >= "2021-01-01"].copy()

print(f"   Price  : {len(price_df):,} rows  ({price_df['Date'].min().date()} → {price_df['Date'].max().date()})")
print(f"   Tweets : {len(tweets_df):,} rows  ({tweets_df['Date'].min().date()} → {tweets_df['Date'].max().date()})")
print(f"   News   : {len(news_df):,} rows  ({news_df['Date'].min().date()} → {news_df['Date'].max().date()})")

# ─── Keep only useful news columns ────────────────────────────────────────────
news_cols_keep = ["Date", "News_Count", "Avg_Sentiment",
                  "Positive_Count", "Negative_Count", "Neutral_Count"]
news_cols_keep = [c for c in news_cols_keep if c in news_df.columns]
news_df = news_df[news_cols_keep].copy()

# Rename Avg_Sentiment to avoid collision
news_df.rename(columns={"Avg_Sentiment": "News_Avg_Sentiment",
                         "News_Count": "News_Count",
                         "Positive_Count": "News_Positive_Count",
                         "Negative_Count": "News_Negative_Count",
                         "Neutral_Count":  "News_Neutral_Count"}, inplace=True)

tweets_df.rename(columns={"Avg_Sentiment": "Tweet_Avg_Sentiment",
                           "Tweet_Count": "Tweet_Count",
                           "Positive_Count": "Tweet_Positive_Count",
                           "Negative_Count": "Tweet_Negative_Count",
                           "Neutral_Count":  "Tweet_Neutral_Count"}, inplace=True)

# ─── Experiment 1: Price + News ───────────────────────────────────────────────
print("\n[2/3] Creating master_price_news.csv (Experiment 1)...")

exp1 = pd.merge(price_df, news_df, on="Date", how="inner")
exp1.sort_values("Date", inplace=True)
exp1.reset_index(drop=True, inplace=True)

sentiment_cols_news = ["News_Avg_Sentiment", "News_Count",
                       "News_Positive_Count", "News_Negative_Count", "News_Neutral_Count"]
sentiment_cols_news = [c for c in sentiment_cols_news if c in exp1.columns]

exp1 = add_features(exp1, sentiment_cols_news)

out1 = os.path.join(BASE_DIR, "master_price_news.csv")
exp1.to_csv(out1, index=False)
print(f"    Saved: master_price_news.csv")
print(f"   Rows  : {len(exp1):,}")
print(f"   Cols  : {len(exp1.columns)}")
print(f"   Range : {exp1['Date'].min().date()} → {exp1['Date'].max().date()}")

# ─── Experiment 2: Price + Tweets ─────────────────────────────────────────────
print("\n[3/3] Creating master_price_tweets.csv (Experiment 2)...")

exp2 = pd.merge(price_df, tweets_df, on="Date", how="inner")
exp2.sort_values("Date", inplace=True)
exp2.reset_index(drop=True, inplace=True)

sentiment_cols_tweets = ["Tweet_Avg_Sentiment", "Tweet_Count",
                         "Tweet_Positive_Count", "Tweet_Negative_Count", "Tweet_Neutral_Count"]
sentiment_cols_tweets = [c for c in sentiment_cols_tweets if c in exp2.columns]

exp2 = add_features(exp2, sentiment_cols_tweets)

out2 = os.path.join(BASE_DIR, "master_price_tweets.csv")
exp2.to_csv(out2, index=False)
print(f"    Saved: master_price_tweets.csv")
print(f"   Rows  : {len(exp2):,}")
print(f"   Cols  : {len(exp2.columns)}")
print(f"   Range : {exp2['Date'].min().date()} → {exp2['Date'].max().date()}")

# ─── Experiment 3: Price + Tweets + News ──────────────────────────────────────
print("\n[4/4] Creating master_dataset.csv (Experiment 3 - All)...")

exp3 = pd.merge(price_df, tweets_df, on="Date", how="inner")
exp3 = pd.merge(exp3, news_df, on="Date", how="inner")
exp3.sort_values("Date", inplace=True)
exp3.reset_index(drop=True, inplace=True)

sentiment_cols_all = sentiment_cols_tweets + sentiment_cols_news
sentiment_cols_all = [c for c in sentiment_cols_all if c in exp3.columns]

exp3 = add_features(exp3, sentiment_cols_all)

out3 = os.path.join(BASE_DIR, "master_dataset.csv")
exp3.to_csv(out3, index=False)
print(f"    Saved: master_dataset.csv")
print(f"   Rows  : {len(exp3):,}")
print(f"   Cols  : {len(exp3.columns)}")
print(f"   Range : {exp3['Date'].min().date()} → {exp3['Date'].max().date()}")

# ─── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"  Exp 1 (Price+News)     : {len(exp1):>5,} rows | {len(exp1.columns)} cols")
print(f"  Exp 2 (Price+Tweets)   : {len(exp2):>5,} rows | {len(exp2.columns)} cols")
print(f"  Exp 3 (Price+T+N)      : {len(exp3):>5,} rows | {len(exp3.columns)} cols")
print("=" * 55)
print("\n   All 3 master datasets saved in Bitcoin/data/")
print(" Next: py generate_all_plots.py\n")

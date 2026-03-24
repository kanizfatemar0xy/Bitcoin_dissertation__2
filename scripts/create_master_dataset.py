import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# ⚙️  PATHS
# ══════════════════════════════════════════════════════════════
BASE        = r"C:\Users\HAROON KHAN\Desktop\bitcoin_volatility_project"
DATA        = BASE + r"\data"
PRICE_FILE  = DATA + r"\1_price_data.csv"
TWEETS_FILE = DATA + r"\2_tweets_data.csv"
NEWS_FILE   = DATA + r"\3_news_data.csv"
OUT_FILE    = DATA + r"\master_dataset.csv"
# ══════════════════════════════════════════════════════════════

print("=" * 55)
print("STEP 1: Loading Files")
print("=" * 55)

price  = pd.read_csv(PRICE_FILE,  parse_dates=["Date"])
tweets = pd.read_csv(TWEETS_FILE, parse_dates=["Date"])
news   = pd.read_csv(NEWS_FILE,   parse_dates=["Date"])

print(f"  Price  : {len(price):,} rows | {price['Date'].min().date()} --> {price['Date'].max().date()}")
print(f"  Tweets : {len(tweets):,} rows | {tweets['Date'].min().date()} --> {tweets['Date'].max().date()}")
print(f"  News   : {len(news):,} rows | {news['Date'].min().date()} --> {news['Date'].max().date()}")

# ── STEP 2: Rename columns to avoid confusion ─────────────────
print("\n" + "=" * 55)
print("STEP 2: Renaming Columns")
print("=" * 55)

tweets = tweets.rename(columns={
    "Avg_Sentiment":  "Tweet_Sentiment",
    "Positive_Count": "Tweet_Positive",
    "Negative_Count": "Tweet_Negative",
    "Neutral_Count":  "Tweet_Neutral",
    "Tweet_Count":    "Tweet_Count"
})

news = news.rename(columns={
    "Avg_Sentiment":  "News_Sentiment",
    "Positive_Count": "News_Positive",
    "Negative_Count": "News_Negative",
    "Neutral_Count":  "News_Neutral",
    "News_Count":     "News_Count"
})

print("  ✅ Columns renamed.")

# ── STEP 3: Merge all three ───────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Merging Datasets")
print("=" * 55)

# Price + Tweets
df = pd.merge(price, tweets[["Date","Tweet_Sentiment","Tweet_Count",
                               "Tweet_Positive","Tweet_Negative","Tweet_Neutral"]],
              on="Date", how="inner")

# + News
df = pd.merge(df, news[["Date","News_Sentiment","News_Count",
                          "News_Positive","News_Negative","News_Neutral"]],
              on="Date", how="inner")

df = df.sort_values("Date").reset_index(drop=True)

print(f"  Merged rows  : {len(df):,}")
print(f"  Date range   : {df['Date'].min().date()} --> {df['Date'].max().date()}")
print(f"  Columns ({df.shape[1]}):")
for col in df.columns:
    print(f"    - {col}")

# ── STEP 4: Feature Engineering ───────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Feature Engineering")
print("=" * 55)

# Lag features (previous days)
for lag in [1, 2, 3, 7]:
    df[f"Close_Lag{lag}"]           = df["Close"].shift(lag)
    df[f"Tweet_Sentiment_Lag{lag}"] = df["Tweet_Sentiment"].shift(lag)
    df[f"News_Sentiment_Lag{lag}"]  = df["News_Sentiment"].shift(lag)
    df[f"Return_Lag{lag}"]          = df["Daily_Return"].shift(lag)

# Rolling features
df["Close_MA7"]          = df["Close"].rolling(7).mean()
df["Close_MA30"]         = df["Close"].rolling(30).mean()
df["Return_Std7"]        = df["Daily_Return"].rolling(7).std()
df["Tweet_Sent_MA7"]     = df["Tweet_Sentiment"].rolling(7).mean()
df["News_Sent_MA7"]      = df["News_Sentiment"].rolling(7).mean()
df["Volume_MA7"]         = df["Volume"].rolling(7).mean()

# Price range feature
df["Price_Range"]        = df["High"] - df["Low"]
df["Price_Range_Pct"]    = (df["High"] - df["Low"]) / df["Close"] * 100

print(f"  ✅ Lag features added (1,2,3,7 days)")
print(f"  ✅ Rolling features added (MA7, MA30, Std7)")
print(f"  ✅ Price range features added")

# ── STEP 5: Target Variable ───────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Target Variable")
print("=" * 55)

# Target = Volatility_7d (already in price data)
print(f"  Target variable  : Volatility_7d")
print(f"  Null in target   : {df['Volatility_7d'].isna().sum()}")

# ── STEP 6: Drop nulls ────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Dropping Nulls")
print("=" * 55)

before = len(df)
df = df.dropna().reset_index(drop=True)
after  = len(df)

print(f"  Rows before : {before}")
print(f"  Rows after  : {after}")
print(f"  Dropped     : {before - after}")
print(f"  Date range  : {df['Date'].min().date()} --> {df['Date'].max().date()}")

# ── STEP 7: Save ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 7: Saving Master Dataset")
print("=" * 55)

df.to_csv(OUT_FILE, index=False)
print(f"  ✅ Saved: data/master_dataset.csv")
print(f"  Shape  : {df.shape}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("MASTER DATASET SUMMARY")
print("=" * 55)
print(f"""
  Total rows     : {len(df):,}
  Total features : {df.shape[1] - 2}  (excluding Date & Target)
  Target         : Volatility_7d
  Date range     : {df['Date'].min().date()} --> {df['Date'].max().date()}

  Feature Groups:
    Price features      : Close, Open, High, Low, Volume, Daily_Return
    Volatility          : Volatility_7d (TARGET), Volatility_30d
    Tweet features      : Tweet_Sentiment, Tweet_Count, Positive/Negative/Neutral
    News features       : News_Sentiment, News_Count, Positive/Negative/Neutral
    Lag features        : Close/Return/Sentiment lags (1,2,3,7 days)
    Rolling features    : MA7, MA30, Std7
    Price range         : Price_Range, Price_Range_Pct

  Numeric Summary (key columns):
""")
key_cols = ["Close", "Daily_Return", "Volatility_7d",
            "Tweet_Sentiment", "News_Sentiment"]
print(df[key_cols].describe().round(4).to_string())

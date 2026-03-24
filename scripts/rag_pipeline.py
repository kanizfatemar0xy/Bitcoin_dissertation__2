import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# ⚙️  PATHS
# ══════════════════════════════════════════════════════════════
BASE        = r"C:\Users\HAROON KHAN\Desktop\bitcoin_volatility_project"
DATA        = BASE + r"\data"
PLOTS_DIR   = BASE + r"\plots\rag"
RESULTS_DIR = BASE + r"\rag_results"
MASTER_FILE = DATA + r"\master_dataset.csv"
NEWS_FILE   = DATA + r"\3_news_data.csv"
TWEETS_FILE = DATA + r"\2_tweets_data.csv"
SHAP_FILE   = DATA + r"\shap_feature_importance.csv"
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
# ══════════════════════════════════════════════════════════════

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor":   "black", "font.family": "DejaVu Sans",
    "axes.titlesize":   11,      "axes.titleweight": "bold",
    "axes.labelsize":   10,      "xtick.labelsize": 9,
    "ytick.labelsize":  9,       "grid.color": "#cccccc",
    "grid.linestyle":   "--",    "grid.alpha": 0.5,
})

# ══════════════════════════════════════════════════════════════
# STEP 1: Load All Data
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

master = pd.read_csv(MASTER_FILE, parse_dates=["Date"])
news   = pd.read_csv(NEWS_FILE,   parse_dates=["Date"])
tweets = pd.read_csv(TWEETS_FILE, parse_dates=["Date"])
shap   = pd.read_csv(SHAP_FILE,   index_col=0)

print(f"  Master : {len(master):,} rows | {master['Date'].min().date()} --> {master['Date'].max().date()}")
print(f"  News   : {len(news):,} rows   | {news['Date'].min().date()} --> {news['Date'].max().date()}")
print(f"  Tweets : {len(tweets):,} rows  | {tweets['Date'].min().date()} --> {tweets['Date'].max().date()}")

# ══════════════════════════════════════════════════════════════
# STEP 2: Identify High Volatility Events
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Identifying High Volatility Events")
print("=" * 60)

threshold = master["Volatility_7d"].quantile(0.90)
high_vol  = master[master["Volatility_7d"] >= threshold].copy()
high_vol  = high_vol.sort_values("Volatility_7d", ascending=False)

print(f"  Volatility threshold (90th pct) : {threshold:.4f}%")
print(f"  High volatility days found      : {len(high_vol)}")
print(f"\n  Top 10 Most Volatile Days:")
print(high_vol[["Date","Close","Daily_Return","Volatility_7d",
                "Tweet_Sentiment","News_Sentiment"]].head(10).to_string(index=False))

# ── Plot 1: Volatility Timeline ───────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(master["Date"], master["Volatility_7d"],
        color="black", linewidth=1.0, label="7-Day Volatility")
ax.axhline(threshold, color="gray", linewidth=1.0,
           linestyle="--", label=f"90th Pct Threshold ({threshold:.2f}%)")
ax.scatter(high_vol["Date"], high_vol["Volatility_7d"],
           color="black", s=25, zorder=5, label="High Volatility Events")
ax.set_title("Bitcoin 7-Day Volatility — High Volatility Events Highlighted")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility (%)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rag_volatility_events.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✅ Saved: plots/rag/rag_volatility_events.png")

# ══════════════════════════════════════════════════════════════
# STEP 3: Retrieval Function
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Building Retrieval System")
print("=" * 60)

def retrieve_context(event_date, window_days=3):
    start      = event_date - timedelta(days=window_days)
    end        = event_date + timedelta(days=1)
    rel_news   = news[(news["Date"] >= start)     & (news["Date"] <= end)].copy()
    rel_tweets = tweets[(tweets["Date"] >= start) & (tweets["Date"] <= end)].copy()
    return rel_news, rel_tweets

print("  ✅ Retrieval function ready.")

# ══════════════════════════════════════════════════════════════
# STEP 4: Narrative Generation
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: RAG — Generating Narratives for Top 5 Events")
print("=" * 60)

top_features = shap["Mean"].head(5).index.tolist()

def generate_narrative(event_row, rel_news, rel_tweets, top_features):
    date       = event_row["Date"].date()
    volatility = event_row["Volatility_7d"]
    price      = event_row["Close"]
    ret        = event_row["Daily_Return"]
    tweet_sent = event_row["Tweet_Sentiment"]
    news_sent  = event_row["News_Sentiment"]
    tweet_dir  = "positive" if tweet_sent > 0 else "negative"
    news_dir   = "positive" if news_sent  > 0 else "negative"
    price_dir  = "increased" if ret > 0   else "decreased"
    n_news     = len(rel_news)
    n_tweets   = len(rel_tweets)
    avg_ns     = rel_news["Avg_Sentiment"].mean()   if n_news   > 0 else 0.0
    avg_ts     = rel_tweets["Avg_Sentiment"].mean() if n_tweets > 0 else 0.0

    narrative = f"""
{'='*62}
HIGH VOLATILITY EVENT — {date}
{'='*62}

QUANTITATIVE ANALYSIS:
  Date              : {date}
  Bitcoin Price     : ${price:,.2f}
  Daily Return      : {ret:+.2f}%
  7-Day Volatility  : {volatility:.4f}%  (above 90th percentile)
  Tweet Sentiment   : {tweet_sent:+.4f}  ({tweet_dir})
  News Sentiment    : {news_sent:+.4f}   ({news_dir})

SHAP — KEY DRIVERS:
  Top features contributing to this volatility spike:
{chr(10).join([f"  • {f}" for f in top_features])}

RETRIEVED CONTEXT:
  News articles (+-3 days)      : {n_news}
  Tweet records (+-3 days)      : {n_tweets}
  Avg News Sentiment (window)   : {avg_ns:+.4f}
  Avg Tweet Sentiment (window)  : {avg_ts:+.4f}

NARRATIVE EXPLANATION:
  On {date}, Bitcoin experienced elevated volatility of {volatility:.2f}%.
  The price {price_dir} by {abs(ret):.2f}%, closing at ${price:,.2f}.

  SHAP analysis identified Return_Std7 (7-day return standard
  deviation) as the dominant driver, indicating that accumulated
  price instability was the strongest predictor of this spike.

  Social media sentiment was {tweet_dir} ({tweet_sent:+.3f}) on this day,
  {'suggesting market optimism despite the volatility.'
   if tweet_sent > 0
   else 'reflecting investor anxiety and negative market sentiment.'}

  News sentiment was {news_dir} ({news_sent:+.3f}),
  {'indicating positive coverage which may have amplified buying pressure.'
   if news_sent > 0
   else 'indicating cautious or negative media coverage around this period.'}

  {'[BEAR EVENT] High volatility with negative return suggests a sell-off, potentially triggered by negative news, FUD, or liquidation cascade.'
   if ret < 0
   else '[BULL EVENT] High volatility with positive return suggests a strong rally, potentially driven by positive news catalysts or FOMO from social media.'}
{'='*62}
"""
    return narrative

top5_events    = high_vol.head(5)
all_narratives = []

for idx, row in top5_events.iterrows():
    event_date           = row["Date"]
    rel_news, rel_tweets = retrieve_context(event_date, window_days=3)
    narrative            = generate_narrative(row, rel_news, rel_tweets, top_features)
    all_narratives.append({
        "date":       str(event_date.date()),
        "volatility": round(float(row["Volatility_7d"]), 4),
        "price":      round(float(row["Close"]), 2),
        "return":     round(float(row["Daily_Return"]), 4),
        "narrative":  narrative
    })
    print(narrative)

# Save narratives
with open(os.path.join(RESULTS_DIR, "rag_narratives.json"), "w") as f:
    json.dump(all_narratives, f, indent=2)
print("  ✅ Saved: rag_results/rag_narratives.json")

with open(os.path.join(RESULTS_DIR, "rag_narratives.txt"), "w", encoding="utf-8") as f:
    for n in all_narratives:
        f.write(n["narrative"])
        f.write("\n\n")
print("  ✅ Saved: rag_results/rag_narratives.txt")

# ══════════════════════════════════════════════════════════════
# STEP 5: Sentiment vs Volatility Plots — FIXED (no overlap)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Sentiment vs Volatility Plots")
print("=" * 60)

fig, axes = plt.subplots(5, 1, figsize=(13, 22))

for i, (idx, row) in enumerate(top5_events.iterrows()):
    event_date = row["Date"]
    start      = event_date - timedelta(days=14)
    end        = event_date + timedelta(days=7)
    window     = master[(master["Date"] >= start) & (master["Date"] <= end)]

    ax1 = axes[i]
    ax2 = ax1.twinx()

    ax1.plot(window["Date"], window["Volatility_7d"],
             color="black", linewidth=1.8, label="Volatility (%)")
    ax2.plot(window["Date"], window["Tweet_Sentiment"],
             color="#555555", linewidth=1.2, linestyle="--",
             label="Tweet Sentiment")
    ax2.plot(window["Date"], window["News_Sentiment"],
             color="#999999", linewidth=1.2, linestyle=":",
             label="News Sentiment")
    ax1.axvline(event_date, color="black", linewidth=1.5,
                linestyle="--", alpha=0.6, label="Event Date")

    # ✅ Fixed title — loc=left, no suptitle
    ax1.set_title(
        f"Event {i+1}:  {event_date.date()}   |   "
        f"Volatility = {row['Volatility_7d']:.2f}%   |   "
        f"Return = {row['Daily_Return']:+.2f}%",
        pad=8, loc="left"
    )
    ax1.set_ylabel("Volatility (%)", fontsize=10)
    ax2.set_ylabel("Sentiment Score", fontsize=10)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha="right")
    ax1.grid(True)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               loc="upper left", fontsize=8, framealpha=0.9)

# ✅ Proper spacing — no overlap
plt.subplots_adjust(top=0.97, hspace=0.55)
plt.savefig(os.path.join(PLOTS_DIR, "rag_event_analysis.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: plots/rag/rag_event_analysis.png")

# ══════════════════════════════════════════════════════════════
# STEP 6: Summary Table
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: RAG Summary Table")
print("=" * 60)

summary = high_vol[["Date","Close","Daily_Return","Volatility_7d",
                     "Tweet_Sentiment","News_Sentiment"]].head(10).copy()
summary["Event_Type"] = summary["Daily_Return"].apply(
    lambda x: "Bull Spike" if x > 0 else "Bear Crash"
)
summary["Sentiment_Aligned"] = (
    (summary["Daily_Return"] > 0) == (summary["Tweet_Sentiment"] > 0)
).map({True: "Yes", False: "No"})

summary.to_csv(os.path.join(RESULTS_DIR, "rag_summary_table.csv"), index=False)
print("\n  Top 10 High Volatility Events:")
print(summary.to_string(index=False))
print("\n  ✅ Saved: rag_results/rag_summary_table.csv")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
bull    = (summary["Event_Type"] == "Bull Spike").sum()
bear    = (summary["Event_Type"] == "Bear Crash").sum()
aligned = (summary["Sentiment_Aligned"] == "Yes").sum()

print("\n" + "=" * 60)
print("RAG PIPELINE COMPLETE")
print("=" * 60)
print(f"""
  High Volatility Events : {len(high_vol)}
  Bull Spikes            : {bull}
  Bear Crashes           : {bear}
  Sentiment Aligned      : {aligned}/10 events

  Saved Plots:
    plots/rag/rag_volatility_events.png
    plots/rag/rag_event_analysis.png

  Saved Results:
    rag_results/rag_narratives.txt
    rag_results/rag_narratives.json
    rag_results/rag_summary_table.csv

  ALL EXPERIMENTATION COMPLETE!
  Next Step: Results Analysis and Report Writing
""")
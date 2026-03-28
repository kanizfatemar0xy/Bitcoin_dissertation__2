"""
Script 5: rag_pipeline.py
===========================
Teeno experiments ke high-volatility events ke liye
RAG (Retrieval-Augmented Generation) narratives banata hai.

Approach:
  1. High volatility events detect karta hai (threshold > mean + 1.5*std)
  2. Un dates ke aas paas ke news/tweet sentiment retrieve karta hai
  3. TF-IDF similarity se most relevant context select karta hai
  4. Claude-style narrative template se explanation generate karta hai

Output:
  rag/
  ├── exp1/
  │   ├── rag_narratives.txt
  │   ├── rag_narratives.json
  │   ├── rag_summary_table.csv
  │   ├── rag_volatility_events.png
  │   └── rag_event_analysis.png
  ├── exp2/  (same)
  └── exp3/  (same)

Run karo:  py rag_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise         import cosine_similarity

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAG_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag")

PRICE_FILE  = os.path.join(BASE_DIR, "1_price_data.csv")
TWEETS_FILE = os.path.join(BASE_DIR, "2_tweets_data.csv")
NEWS_FILE   = os.path.join(BASE_DIR, "3_news_data.csv")

EXPERIMENTS = {
    "exp1": {
        "file"         : os.path.join(BASE_DIR, "master_price_news.csv"),
        "label"        : "Experiment 1 — Price + News",
        "has_tweets"   : False,
        "has_news"     : True,
    },
    "exp2": {
        "file"         : os.path.join(BASE_DIR, "master_price_tweets.csv"),
        "label"        : "Experiment 2 — Price + Tweets",
        "has_tweets"   : True,
        "has_news"     : False,
    },
    "exp3": {
        "file"         : os.path.join(BASE_DIR, "master_dataset.csv"),
        "label"        : "Experiment 3 — Price + Tweets + News",
        "has_tweets"   : True,
        "has_news"     : True,
    },
}

TARGET    = "Volatility_7d"
TOP_N     = 10     # top N high-volatility events
WINDOW    = 3      # days before/after event for context
BG        = "#FAFAFA"
C_PRICE   = "#F4A825"
C_VOL     = "#E74C3C"
C_TWEETS  = "#1DA1F2"
C_NEWS    = "#2ECC71"

# ─── Known Bitcoin market events (for narrative context) ──────────────────────
KNOWN_EVENTS = {
    "2021-01": "Bitcoin bull run — BTC crossed $40K for first time",
    "2021-02": "Tesla announced $1.5B Bitcoin investment",
    "2021-04": "Bitcoin ATH near $65K — Coinbase IPO",
    "2021-05": "China crypto mining ban — massive crash",
    "2021-06": "El Salvador Bitcoin legal tender announcement",
    "2021-07": "Bitcoin recovery after China ban",
    "2021-09": "El Salvador officially adopted Bitcoin",
    "2021-10": "First Bitcoin ETF (ProShares) approved",
    "2021-11": "Bitcoin new ATH near $69K",
    "2021-12": "Bitcoin year-end correction",
    "2022-01": "Bitcoin correction — Fed rate hike fears",
    "2022-03": "Russia-Ukraine war — market uncertainty",
    "2022-05": "Terra/LUNA collapse — crypto market crash",
    "2022-06": "Celsius Network froze withdrawals — bear market",
    "2022-07": "Crypto market partial recovery",
    "2022-09": "Ethereum Merge — market volatility",
    "2022-11": "FTX collapse — major crypto crisis",
    "2022-12": "Post-FTX bear market bottom",
    "2023-01": "Bitcoin recovery from FTX crash lows",
    "2023-02": "Bitcoin consolidation phase",
    "2023-03": "Silvergate/SVB banking crisis — BTC spike",
    "2023-06": "BlackRock Bitcoin ETF application",
    "2023-10": "Bitcoin ETF speculation — price surge",
    "2023-12": "Year-end rally — ETF approval anticipation",
    "2024-01": "Bitcoin Spot ETF approved by SEC",
    "2024-02": "Post-ETF consolidation",
    "2024-03": "Bitcoin new ATH before halving",
    "2024-04": "Bitcoin halving event",
    "2024-05": "Post-halving market reaction",
    "2024-06": "Summer consolidation",
    "2024-07": "Market recovery",
    "2024-08": "Macro uncertainty — Fed rate decisions",
    "2024-09": "September correction",
}

def get_event_context(date):
    """Date ke liye known event context dhundho."""
    key = date.strftime("%Y-%m")
    return KNOWN_EVENTS.get(key, "General market activity period")

def sentiment_label(score):
    if score > 0.1:  return "Positive 📈"
    if score < -0.1: return "Negative 📉"
    return "Neutral ➡️"

def volatility_level(vol, mean_vol, std_vol):
    if vol > mean_vol + 2*std_vol: return "EXTREME"
    if vol > mean_vol + 1.5*std_vol: return "HIGH"
    if vol > mean_vol + std_vol: return "ELEVATED"
    return "MODERATE"

# ─── RAG: Simple TF-IDF retrieval ─────────────────────────────────────────────
def build_context_corpus(df, event_date, window=3, has_tweets=True, has_news=True):
    """Event ke aas paas ke rows se context corpus banao."""
    start = event_date - pd.Timedelta(days=window)
    end   = event_date + pd.Timedelta(days=window)
    mask  = (df["Date"] >= start) & (df["Date"] <= end)
    ctx   = df[mask].copy()

    docs = []
    for _, row in ctx.iterrows():
        parts = [f"Date: {row['Date'].date()}"]
        parts.append(f"Close Price: ${row['Close']:,.0f}")
        parts.append(f"Daily Return: {row.get('Daily_Return', 0)*100:.2f}%")
        parts.append(f"Volatility: {row[TARGET]:.6f}")

        if has_tweets and "Tweet_Avg_Sentiment" in row:
            parts.append(f"Tweet Sentiment: {row['Tweet_Avg_Sentiment']:.3f} ({sentiment_label(row['Tweet_Avg_Sentiment'])})")
        if has_news and "News_Avg_Sentiment" in row:
            parts.append(f"News Sentiment: {row['News_Avg_Sentiment']:.3f} ({sentiment_label(row['News_Avg_Sentiment'])})")

        docs.append(" | ".join(parts))

    return docs

def rag_retrieve(query, corpus, top_k=3):
    """TF-IDF se most relevant documents retrieve karo."""
    if not corpus:
        return []
    all_docs = [query] + corpus
    try:
        tfidf = TfidfVectorizer(stop_words="english", min_df=1)
        matrix = tfidf.fit_transform(all_docs)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [corpus[i] for i in top_idx]
    except:
        return corpus[:top_k]

def generate_narrative(event, exp_label, mean_vol, std_vol, has_tweets, has_news):
    """Event ke liye structured narrative generate karo."""
    date      = event["date"]
    vol       = event["volatility"]
    price     = event["price"]
    ret       = event["daily_return"]
    vol_lvl   = volatility_level(vol, mean_vol, std_vol)
    context   = get_event_context(date)
    retrieved = event.get("retrieved_docs", [])

    lines = []
    lines.append(f"{'='*65}")
    lines.append(f"EVENT #{event['rank']} — {date.strftime('%B %d, %Y')}")
    lines.append(f"{'='*65}")
    lines.append(f"Volatility Level  : {vol_lvl} ({vol:.6f})")
    lines.append(f"Bitcoin Price     : ${price:,.2f}")
    lines.append(f"Daily Return      : {ret*100:+.2f}%")
    lines.append(f"")
    lines.append(f"[MARKET CONTEXT]")
    lines.append(f"  {context}")
    lines.append(f"")

    if has_tweets and event.get("tweet_sentiment") is not None:
        ts = event["tweet_sentiment"]
        lines.append(f"[SOCIAL MEDIA SIGNAL]")
        lines.append(f"  Twitter Sentiment : {ts:.3f} — {sentiment_label(ts)}")
        if ts > 0.1:
            lines.append(f"  → Positive social media may have amplified buying pressure.")
        elif ts < -0.1:
            lines.append(f"  → Negative sentiment likely contributed to selling pressure.")
        else:
            lines.append(f"  → Neutral social media — volatility driven by other factors.")
        lines.append(f"")

    if has_news and event.get("news_sentiment") is not None:
        ns = event["news_sentiment"]
        lines.append(f"[NEWS SIGNAL]")
        lines.append(f"  News Sentiment    : {ns:.3f} — {sentiment_label(ns)}")
        if ns > 0.1:
            lines.append(f"  → Positive news coverage may have driven price upward.")
        elif ns < -0.1:
            lines.append(f"  → Negative news likely triggered fear-based selling.")
        else:
            lines.append(f"  → Mixed news signals — institutional/macro factors dominant.")
        lines.append(f"")

    lines.append(f"[RETRIEVED CONTEXT — RAG]")
    if retrieved:
        for i, doc in enumerate(retrieved, 1):
            lines.append(f"  [{i}] {doc}")
    else:
        lines.append(f"  No close context records found.")
    lines.append(f"")

    lines.append(f"[ANALYSIS SUMMARY]")
    if vol_lvl in ("EXTREME", "HIGH"):
        lines.append(f"  This was a significant volatility spike ({vol_lvl}). "
                     f"The {abs(ret)*100:.1f}% price {'gain' if ret>0 else 'drop'} "
                     f"on this date aligns with: {context}.")
    else:
        lines.append(f"  Elevated but not extreme volatility. Market showed "
                     f"{'upward' if ret>0 else 'downward'} movement of {abs(ret)*100:.1f}%.")
    lines.append(f"")

    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  RAG PIPELINE — ALL EXPERIMENTS")
print("=" * 60)

for exp_key, exp_info in EXPERIMENTS.items():

    print(f"\n{'─'*60}")
    print(f"  {exp_info['label']}")
    print(f"{'─'*60}")

    if not os.path.exists(exp_info["file"]):
        print(f"  ⚠  Skipped — file not found: {exp_info['file']}")
        continue

    df = pd.read_csv(exp_info["file"], parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    has_tweets = exp_info["has_tweets"] and "Tweet_Avg_Sentiment" in df.columns
    has_news   = exp_info["has_news"]   and "News_Avg_Sentiment"  in df.columns

    mean_vol = df[TARGET].mean()
    std_vol  = df[TARGET].std()
    threshold = mean_vol + 1.5 * std_vol

    high_vol = df[df[TARGET] > threshold].copy()
    high_vol = high_vol.sort_values(TARGET, ascending=False).head(TOP_N)
    high_vol = high_vol.reset_index(drop=True)

    print(f"  Rows          : {len(df):,}")
    print(f"  Mean Vol      : {mean_vol:.6f}")
    print(f"  Threshold     : {threshold:.6f}  (mean + 1.5σ)")
    print(f"  High Vol Days : {len(df[df[TARGET] > threshold])}")
    print(f"  Top events    : {len(high_vol)}")

    out_dir = os.path.join(RAG_DIR, exp_key)
    os.makedirs(out_dir, exist_ok=True)

    # ── Build events list ─────────────────────────────────────
    events = []
    for rank, (_, row) in enumerate(high_vol.iterrows(), 1):
        date = row["Date"]
        corpus = build_context_corpus(df, date, WINDOW, has_tweets, has_news)
        query  = f"high volatility bitcoin {date.strftime('%Y %B')} price movement"
        retrieved = rag_retrieve(query, corpus, top_k=3)

        ev = {
            "rank"          : rank,
            "date"          : date,
            "volatility"    : row[TARGET],
            "price"         : row["Close"],
            "daily_return"  : row.get("Daily_Return", 0),
            "retrieved_docs": retrieved,
            "tweet_sentiment": row.get("Tweet_Avg_Sentiment") if has_tweets else None,
            "news_sentiment" : row.get("News_Avg_Sentiment")  if has_news   else None,
        }
        events.append(ev)

    # ── Generate narratives ───────────────────────────────────
    print(f"\n  Generating narratives...")
    all_narratives = []
    header = f"RAG VOLATILITY ANALYSIS — {exp_info['label'].upper()}\n"
    header += f"Generated for top {TOP_N} high-volatility events\n"
    header += "=" * 65 + "\n\n"
    all_narratives.append(header)

    for ev in events:
        narr = generate_narrative(ev, exp_info["label"], mean_vol, std_vol, has_tweets, has_news)
        all_narratives.append(narr)

    # ── Save TXT ──────────────────────────────────────────────
    txt_path = os.path.join(out_dir, "rag_narratives.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_narratives))
    print(f"   ✅ rag_narratives.txt")

    # ── Save JSON ─────────────────────────────────────────────
    json_data = []
    for ev in events:
        json_data.append({
            "rank"           : ev["rank"],
            "date"           : ev["date"].strftime("%Y-%m-%d"),
            "volatility"     : round(ev["volatility"], 6),
            "price"          : round(ev["price"], 2),
            "daily_return"   : round(ev.get("daily_return", 0), 4),
            "vol_level"      : volatility_level(ev["volatility"], mean_vol, std_vol),
            "market_context" : get_event_context(ev["date"]),
            "tweet_sentiment": round(ev["tweet_sentiment"], 4) if ev["tweet_sentiment"] is not None else None,
            "news_sentiment" : round(ev["news_sentiment"],  4) if ev["news_sentiment"]  is not None else None,
            "retrieved_docs" : ev["retrieved_docs"],
        })

    json_path = os.path.join(out_dir, "rag_narratives.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"   ✅ rag_narratives.json")

    # ── Save CSV summary ──────────────────────────────────────
    summary_rows = []
    for ev in events:
        summary_rows.append({
            "Rank"            : ev["rank"],
            "Date"            : ev["date"].strftime("%Y-%m-%d"),
            "Volatility"      : round(ev["volatility"], 6),
            "Price_USD"       : round(ev["price"], 2),
            "Daily_Return_Pct": round(ev.get("daily_return", 0) * 100, 2),
            "Vol_Level"       : volatility_level(ev["volatility"], mean_vol, std_vol),
            "Market_Context"  : get_event_context(ev["date"]),
            "Tweet_Sentiment" : round(ev["tweet_sentiment"], 4) if ev["tweet_sentiment"] is not None else "N/A",
            "News_Sentiment"  : round(ev["news_sentiment"],  4) if ev["news_sentiment"]  is not None else "N/A",
        })

    csv_path = os.path.join(out_dir, "rag_summary_table.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"   ✅ rag_summary_table.csv")

    # ── Plot 1: Volatility timeline ────────────────────────────
    print(f"\n  Generating plots...")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["Date"], df[TARGET], color=C_VOL, linewidth=1.2, label="Volatility (7d)", alpha=0.8)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1,
               label=f"Threshold ({threshold:.4f})", alpha=0.7)
    # Mark top events
    for ev in events:
        ax.axvline(ev["date"], color="#E74C3C", linewidth=0.7, alpha=0.4)
        ax.scatter(ev["date"], ev["volatility"], color="#E74C3C", s=25, zorder=5)

    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_title(f"Bitcoin Volatility — High-Volatility Events\n({exp_info['label']})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Volatility (7d)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "rag_volatility_events.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ rag_volatility_events.png")

    # ── Plot 2: Top events bar chart ───────────────────────────
    top5 = events[:5]
    labels = [ev["date"].strftime("%Y-%m-%d") for ev in top5]
    vols   = [ev["volatility"] for ev in top5]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, vols, color=C_VOL, alpha=0.85)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_title(f"Top 5 Highest Volatility Events\n({exp_info['label']})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Volatility (7d)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, ev in zip(bars, top5):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0002,
                f"${ev['price']:,.0f}",
                ha="center", va="bottom", fontsize=8, color="#333")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "rag_event_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ rag_event_analysis.png")

    print(f"\n  ✅ {exp_key} RAG done!")

# ─── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL RAG ANALYSIS COMPLETE!")
print("=" * 60)
print("""
  rag/
  ├── exp1/
  │   ├── rag_narratives.txt
  │   ├── rag_narratives.json
  │   ├── rag_summary_table.csv
  │   ├── rag_volatility_events.png
  │   └── rag_event_analysis.png
  ├── exp2/  (same)
  └── exp3/  (same)
""")

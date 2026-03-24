import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# ⚙️  PATHS
# ══════════════════════════════════════════════════════════════
BASE        = r"C:\Users\HAROON KHAN\Desktop\bitcoin_volatility_project"
DATA        = BASE + r"\data"
PLOTS_I     = BASE + r"\plots\individual"
PLOTS_S     = BASE + r"\plots\similarity"

PRICE_FILE  = DATA + r"\1_price_data.csv"
TWEETS_FILE = DATA + r"\2_tweets_data.csv"
NEWS_FILE   = DATA + r"\3_news_data.csv"
# ══════════════════════════════════════════════════════════════

# ── Style — exactly like sample ───────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "text.color":        "black",
    "grid.color":        "#dddddd",
    "grid.linestyle":    "-",
    "grid.alpha":        1.0,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "legend.framealpha": 1.0,
    "legend.edgecolor":  "#cccccc",
})

# ── Load ──────────────────────────────────────────────────────
print("Loading datasets...")
price  = pd.read_csv(PRICE_FILE,  parse_dates=["Date"])
tweets = pd.read_csv(TWEETS_FILE, parse_dates=["Date"])
news   = pd.read_csv(NEWS_FILE,   parse_dates=["Date"])
print("  Done.")

# ══════════════════════════════════════════════════════════════
# PLOT 1 — Bitcoin Close Price
# ══════════════════════════════════════════════════════════════
print("\nPlot 1: Bitcoin Close Price...")

fig, ax = plt.subplots(figsize=(13, 5))
fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.12)

ax.plot(price["Date"], price["Close"],
        color="#E8A838", linewidth=1.2, label="Close Price (USD)")

ax.set_title("Bitcoin Daily Close Price (2021–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, which="both", axis="both")
ax.legend(loc="upper left")
ax.set_xlim(price["Date"].min(), price["Date"].max())

plt.savefig(PLOTS_I + r"\plot1_price.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ plot1_price.png")

# ══════════════════════════════════════════════════════════════
# PLOT 2 — Twitter Sentiment
# ══════════════════════════════════════════════════════════════
print("Plot 2: Twitter Sentiment...")

roll_tw = tweets.set_index("Date")["Avg_Sentiment"].rolling("30D").mean().reset_index()

fig, ax = plt.subplots(figsize=(13, 5))
fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.12)

ax.plot(tweets["Date"], tweets["Avg_Sentiment"],
        color="#aaaaaa", linewidth=0.8, alpha=0.7, label="Daily Sentiment")
ax.plot(roll_tw["Date"], roll_tw["Avg_Sentiment"],
        color="#E8A838", linewidth=1.5, label="30-Day Rolling Avg")
ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")

ax.set_title("Bitcoin Twitter Sentiment Score (2021–2024)")
ax.set_xlabel("Date")
ax.set_ylabel("Sentiment Score (-1 to +1)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True)
ax.legend(loc="upper left")
ax.set_xlim(tweets["Date"].min(), tweets["Date"].max())

plt.savefig(PLOTS_I + r"\plot2_tweets.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ plot2_tweets.png")

# ══════════════════════════════════════════════════════════════
# PLOT 3 — News Sentiment
# ══════════════════════════════════════════════════════════════
print("Plot 3: News Sentiment...")

roll_nw = news.set_index("Date")["Avg_Sentiment"].rolling("30D").mean().reset_index()

fig, ax = plt.subplots(figsize=(13, 5))
fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.15)

ax.plot(news["Date"], news["Avg_Sentiment"],
        color="#aaaaaa", linewidth=0.8, alpha=0.7, label="Daily Sentiment")
ax.plot(roll_nw["Date"], roll_nw["Avg_Sentiment"],
        color="#E8A838", linewidth=1.5, label="30-Day Rolling Avg")
ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")

ax.set_title("Bitcoin News Sentiment Score (2021–2023)")
ax.set_xlabel("Date")
ax.set_ylabel("Sentiment Score (-1 to +1)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.grid(True)
ax.legend(loc="upper left")
ax.set_xlim(news["Date"].min(), news["Date"].max())

plt.savefig(PLOTS_I + r"\plot3_news.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ plot3_news.png")

# ══════════════════════════════════════════════════════════════
# PLOT 4 — Similarity: Price vs Twitter Sentiment
# ══════════════════════════════════════════════════════════════
print("Plot 4: Similarity Price vs Tweets...")

merged_pt = pd.merge(
    price[["Date", "Close", "Daily_Return"]],
    tweets[["Date", "Avg_Sentiment"]],
    on="Date", how="inner"
).sort_values("Date").reset_index(drop=True)

corr_pt = merged_pt["Close"].corr(merged_pt["Avg_Sentiment"])
ret_pt  = merged_pt["Daily_Return"].corr(merged_pt["Avg_Sentiment"])
roll_pt = merged_pt.set_index("Date")["Avg_Sentiment"].rolling("30D").mean().reset_index()

fig, ax1 = plt.subplots(figsize=(13, 5))
fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.12)

ax2 = ax1.twinx()

ax1.plot(merged_pt["Date"], merged_pt["Close"],
         color="#E8A838", linewidth=1.3, label="Bitcoin Price (USD)")
ax2.plot(roll_pt["Date"], roll_pt["Avg_Sentiment"],
         color="#333333", linewidth=1.3, linestyle="--",
         label="30-Day Rolling Sentiment")
ax2.axhline(0, color="#999999", linewidth=0.6, linestyle=":")

ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
ax2.set_ylabel("Sentiment Score (-1 to +1)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.set_title(
    f"Bitcoin Price vs Twitter Sentiment  |  "
    f"Pearson r = {corr_pt:.3f}  |  Return r = {ret_pt:.3f}  |  n = {len(merged_pt)} days"
)
ax1.grid(True)
ax1.set_xlim(merged_pt["Date"].min(), merged_pt["Date"].max())

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")

plt.savefig(PLOTS_S + r"\plot4_price_vs_tweets.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ plot4_price_vs_tweets.png")

# ══════════════════════════════════════════════════════════════
# PLOT 5 — Similarity: Price vs News Sentiment
# ══════════════════════════════════════════════════════════════
print("Plot 5: Similarity Price vs News...")

merged_pn = pd.merge(
    price[["Date", "Close", "Daily_Return"]],
    news[["Date", "Avg_Sentiment"]],
    on="Date", how="inner"
).sort_values("Date").reset_index(drop=True)

corr_pn = merged_pn["Close"].corr(merged_pn["Avg_Sentiment"])
ret_pn  = merged_pn["Daily_Return"].corr(merged_pn["Avg_Sentiment"])
roll_pn = merged_pn.set_index("Date")["Avg_Sentiment"].rolling("30D").mean().reset_index()

fig, ax1 = plt.subplots(figsize=(13, 5))
fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.15)

ax2 = ax1.twinx()

ax1.plot(merged_pn["Date"], merged_pn["Close"],
         color="#E8A838", linewidth=1.3, label="Bitcoin Price (USD)")
ax2.plot(roll_pn["Date"], roll_pn["Avg_Sentiment"],
         color="#333333", linewidth=1.3, linestyle="--",
         label="30-Day Rolling Sentiment")
ax2.axhline(0, color="#999999", linewidth=0.6, linestyle=":")

ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
ax2.set_ylabel("News Sentiment Score (-1 to +1)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax1.set_title(
    f"Bitcoin Price vs News Sentiment  |  "
    f"Pearson r = {corr_pn:.3f}  |  Return r = {ret_pn:.3f}  |  n = {len(merged_pn)} days"
)
ax1.grid(True)
ax1.set_xlim(merged_pn["Date"].min(), merged_pn["Date"].max())

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")

plt.savefig(PLOTS_S + r"\plot5_price_vs_news.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ plot5_price_vs_news.png")

print("\n" + "=" * 55)
print("ALL 5 PLOTS SAVED ✅")
print("=" * 55)
# Bitcoin Volatility Prediction Using News & Social Media with Explainable AI and Retrieval-Augmented Generation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**Masters Research Project — COM748**  
*Kaniz Fatema Roxy · Supervisor: Dr Nasir Iqbal*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Architecture](#project-architecture)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Models](#models)
- [Explainable AI — SHAP](#explainable-ai--shap)
- [Retrieval-Augmented Generation — RAG](#retrieval-augmented-generation--rag)
- [Web Dashboard](#web-dashboard)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [References](#references)

---

## Overview

This project presents an **interpretable, end-to-end framework** for predicting and explaining Bitcoin price volatility. It combines:

- **Machine Learning & Deep Learning** models trained on financial and sentiment data
- **Explainable AI (XAI)** via SHAP (SHapley Additive exPlanations) to identify the most influential features driving volatility
- **Retrieval-Augmented Generation (RAG)** to provide human-readable narrative explanations for high-volatility events
- **A Flask Web Dashboard** for live and historical volatility prediction with interactive model selection

The framework goes beyond traditional black-box prediction — it tells *why* Bitcoin is volatile, not just *when*.

---

## Problem Statement

Bitcoin exhibits extreme price volatility, creating significant risk for investors and financial analysts. Existing AI models for volatility prediction largely operate as black-box systems with limited transparency, which undermines trust and restricts informed decision-making.

This project bridges the gap by building a unified framework that:

1. Predicts Bitcoin volatility using historical price data combined with news and social media sentiment
2. Explains model predictions at the feature level using SHAP
3. Contextualises volatility spikes with narrative summaries using RAG

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  Yahoo Finance (Price) · Kaggle Tweets · Kaggle News        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                        │
│  Daily Returns · Rolling Volatility (7d/30d)                │
│  Lag Features (1,2,3,7 days) · Moving Averages             │
│  Sentiment Aggregation (TextBlob) · Price Range Features    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
            Exp 1       Exp 2       Exp 3
         Price+News  Price+Tweets  All Three
                 │          │          │
                 └──────────┼──────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODEL LAYER                              │
│  Random Forest · XGBoost · Gradient Boosting · SVR         │
│  LSTM · Neural Network                                      │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────┐            ┌─────────────────────────────┐
│   SHAP (XAI)     │            │     RAG PIPELINE            │
│  TreeExplainer   │            │  Event Detection            │
│  KernelExplainer │            │  TF-IDF Retrieval           │
│  Feature Rankings│            │  Narrative Generation       │
└──────────────────┘            └─────────────────────────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
           ┌──────────────────────────────┐
           │     FLASK WEB DASHBOARD      │
           │  Live Prediction · SHAP Plot │
           │  RAG Summary · RMSE/MAE      │
           └──────────────────────────────┘
```

---

## Datasets

Three publicly available datasets were used, all filtered to **2021 onwards** for modern relevance:

| # | Dataset | Source | Raw Rows | After Filter | Date Range |
|---|---------|--------|----------|-------------|------------|
| 1 | Bitcoin Historical Price | Yahoo Finance via Kaggle | 3,810 | 1,512 | 2021-01-01 → 2025-02-20 |
| 2 | Bitcoin Tweets Sentiment | Kaggle | 11,295 (individual) | 887 (daily) | 2021-11-05 → 2024-09-12 |
| 3 | Bitcoin News Sentiment | Kaggle (BTC.csv) | 1,826 | 785 | 2021-01-01 → 2023-02-24 |

**Price features computed:** `Daily_Return`, `Volatility_7d`, `Volatility_30d`, `MA7`, `MA30`, `Return_Std7`, `Price_Range`

**Sentiment features:** `Avg_Sentiment` (TextBlob, −1 to +1), `Tweet_Count` / `News_Count`, `Positive_Count`, `Negative_Count`, `Neutral_Count`

**Lag features added:** 1, 2, 3, 7-day lags for all price and sentiment columns

**Target variable:** `Volatility_7d` (7-day rolling standard deviation of daily returns)

---

## Experiments

Three separate datasets were constructed via inner joins to enable controlled comparison:

| Experiment | Files Merged | Rows | Features | Date Range | Coverage |
|------------|-------------|------|----------|-----------|----------|
| **Exp 1** — Price + News | Price ∩ News | ~755 | ~47 | 2021-01-31 → 2023-02-24 | ~26 months |
| **Exp 2** — Price + Tweets | Price ∩ Tweets | ~858 | ~47 | 2021-12-18 → 2024-09-12 | ~34 months |
| **Exp 3** — Price + Tweets + News | Price ∩ Tweets ∩ News | 429 | 43+ | 2021-12-18 → 2023-02-24 | ~14 months |

> All experiments use 80/20 chronological train/test split to prevent data leakage.

---

## Models

Six models were trained per experiment (18 total):

| Model | Type | Key Hyperparameters |
|-------|------|---------------------|
| **Random Forest** | Ensemble (Tree) | n_estimators=200, random_state=42 |
| **XGBoost** | Gradient Boosting (Tree) | n_estimators=200, lr=0.05, max_depth=6 |
| **Gradient Boosting** | Ensemble (Tree) | n_estimators=200, lr=0.05, max_depth=4 |
| **SVR** | Support Vector | kernel=rbf, C=10, gamma=scale |
| **LSTM** | Deep Learning (RNN) | 64→32 units, Dropout=0.2, timesteps=10 |
| **Neural Network** | Deep Learning (MLP) | 128→64→32 units, Dropout=0.2 |

All models use `StandardScaler` on both features and target. Early stopping (patience=10) is applied for LSTM and NN.

---

## Explainable AI — SHAP

SHAP (SHapley Additive exPlanations) values are computed for all six models using the most appropriate explainer per model type:

| Model | SHAP Explainer | Output |
|-------|---------------|--------|
| Random Forest | `TreeExplainer` | Bar + Beeswarm plots |
| XGBoost | `TreeExplainer` | Bar + Beeswarm plots |
| Gradient Boosting | `TreeExplainer` | Bar + Beeswarm plots |
| SVR | `KernelExplainer` | Bar plot |
| Neural Network | `KernelExplainer` | Bar plot |
| LSTM | `KernelExplainer` (sequence) | Bar plot (avg over timesteps) |

**Key SHAP finding (Exp 3):** `Return_Std7` is the dominant feature (SHAP = 1.26), followed by price lag features. Tweet sentiment ranks 7th, confirming its secondary but meaningful contribution to volatility prediction.

All SHAP results saved to: `shap/exp1/`, `shap/exp2/`, `shap/exp3/`

---

## Retrieval-Augmented Generation — RAG

The RAG pipeline contextualises the top 10 highest-volatility events per experiment:

1. **Event Detection** — Volatility threshold = mean + 1.5σ
2. **Context Retrieval** — TF-IDF cosine similarity retrieves the 3 most relevant surrounding data rows (±3 days)
3. **Narrative Generation** — Structured narrative linking price movement, sentiment signals, and known market events (e.g., FTX collapse, Bitcoin ETF approval, Terra/LUNA crash)

**Output per experiment:**
- `rag_narratives.txt` — human-readable event reports
- `rag_narratives.json` — structured data for the dashboard
- `rag_summary_table.csv` — top 10 events with metadata
- `rag_volatility_events.png` — timeline with marked events
- `rag_event_analysis.png` — top 5 events bar chart

**Key RAG finding:** 5 of the top 10 high-volatility events aligned with negative news sentiment, confirming the news-volatility relationship proposed in the literature.

---

## Web Dashboard

A Flask-based interactive dashboard provides real-time and historical analysis:

### Features

| Feature | Description |
|---------|-------------|
| **Experiment Selection** | Choose Exp 1, 2, or 3 from dropdown |
| **Date Selection** | Any date — uses nearest dataset row if outside range |
| **Model Selection** | All 6 models available per experiment |
| **Live Mode** | Today/yesterday → yfinance + RSS news feeds (Google News, CoinDesk) |
| **Historical Mode** | CSV lookup → exact row or nearest-neighbour |
| **Predicted Volatility** | Next-day volatility forecast |
| **Trend Indicator** | Increase / Decrease / Stable |
| **RMSE / MAE / R²** | Model accuracy metrics shown inline |
| **RAG Summary** | Nearest high-volatility event narrative |
| **SHAP Plot** | Feature impact visualisation per model |

### Live Data Sources
- **Price:** Yahoo Finance via `yfinance`
- **News Sentiment:** Google News RSS + CoinDesk RSS + TextBlob

---

## Project Structure

```
Bitcoin/
│
├── data/
│   ├── 1_price_data.csv              # Raw price data (2021–2025)
│   ├── 2_tweets_data.csv             # Raw tweet sentiment (daily)
│   ├── 3_news_data.csv               # Raw news sentiment (daily)
│   ├── master_price_news.csv         # Exp 1 dataset (~755 rows)
│   ├── master_price_tweets.csv       # Exp 2 dataset (~858 rows)
│   └── master_dataset.csv            # Exp 3 dataset (429 rows)
│
├── models/
│   ├── exp1/                         # Exp 1 trained models + scalers
│   │   ├── random_forest_model.pkl
│   │   ├── xgboost_model.pkl
│   │   ├── gradient_boosting_model.pkl
│   │   ├── svr_model.pkl
│   │   ├── lstm_model.keras
│   │   ├── nn_model.keras
│   │   ├── scaler.pkl
│   │   ├── y_scaler.pkl
│   │   └── results.csv
│   ├── exp2/                         # Exp 2 (same structure)
│   └── exp3/                         # Exp 3 (same structure)
│
├── plots/
│   ├── individual/                   # Price, Tweet, News line plots
│   ├── similarity/                   # Normalized comparison plots
│   └── experiments/
│       ├── exp1_price_news/
│       ├── exp2_price_tweets/
│       └── exp3_all/
│
├── shap/
│   ├── exp1/                         # SHAP bar, beeswarm, comparison
│   ├── exp2/
│   └── exp3/
│
├── rag/
│   ├── exp1/                         # RAG narratives, JSON, CSV, plots
│   ├── exp2/
│   └── exp3/
│
├── results/
│   ├── exp1_results.csv              # R², RMSE, MAE per model
│   ├── exp2_results.csv
│   ├── exp3_results.csv
│   ├── all_experiments_comparison.csv
│   └── plots/
│       ├── exp1_metrics_comparison.png
│       ├── exp2_metrics_comparison.png
│       ├── exp3_metrics_comparison.png
│       ├── all_r2_comparison.png
│       ├── all_rmse_comparison.png
│       ├── all_mae_comparison.png
│       └── actual_vs_predicted/      # 18 plots (6 models × 3 experiments)
│
├── scripts/
│   ├── create_master_datasets.py     # Step 1 — build 3 master datasets
│   ├── generate_all_plots.py         # Step 2 — all EDA plots
│   ├── train_all_experiments.py      # Step 3 — train 18 models
│   ├── shap_analysis.py             # Step 4 — SHAP for all models
│   ├── rag_pipeline.py              # Step 5 — RAG narratives
│   └── evaluate_all_experiments.py  # Step 6 — RMSE/MAE evaluation
│
├── templates/
│   ├── index.html                    # Dashboard home page
│   └── results.html                  # Prediction results page
│
├── app.py                            # Flask web application
└── README.md                         # This file
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone / Download the project

```bash
git clone https://github.com/yourusername/bitcoin-volatility-xai.git
cd bitcoin-volatility-xai
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost tensorflow shap flask yfinance textblob
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

<details>
<summary>Full requirements list</summary>

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
xgboost>=2.0
tensorflow>=2.13
shap>=0.43
flask>=3.0
yfinance>=0.2
textblob>=0.17
```

</details>

### 3. Place raw data files

Download datasets from Kaggle and place in `data/`:

| File | Kaggle Link |
|------|-------------|
| `1_price_data.csv` | [Bitcoin Historical Data 2014–2025](https://www.kaggle.com/datasets/eldintarofarrandi/bitcoin-historical-data-2014-2025-yahoo-finance) |
| `2_tweets_data.csv` | [Bitcoin Tweets](https://www.kaggle.com/datasets/sujaykapadnis/bitcoin-tweets) |
| `3_news_data.csv` | [Bitcoin News Dataset](https://www.kaggle.com/datasets/ashirwadsangwan/bitcoinnews-dataset) |

---

## How to Run

Run the scripts **in order** from the `scripts/` folder:

```bash
cd Bitcoin

# Step 1 — Create 3 master datasets (~2 min)
py scripts/create_master_datasets.py

# Step 2 — Generate all EDA plots (~1 min)
py scripts/generate_all_plots.py

# Step 3 — Train all 18 models (~30–45 min)
py scripts/train_all_experiments.py

# Step 4 — SHAP analysis for all models (~20–30 min)
py scripts/shap_analysis.py

# Step 5 — RAG pipeline (~2 min)
py scripts/rag_pipeline.py

# Step 6 — Evaluate RMSE / MAE / R² (~5–10 min)
py scripts/evaluate_all_experiments.py

# Step 7 — Launch web dashboard
py app.py
```

Then open your browser: **http://127.0.0.1:5000**

---

## Results Summary

### Model Performance — Experiment 3 (Price + Tweets + News)

> Best overall results achieved on the combined dataset.

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Random Forest | 0.9853 | — | — |
| XGBoost | 0.9906 | — | — |
| Gradient Boosting | 0.9941 | — | — |
| **SVR** | **0.9985** | — | — |
| LSTM | −0.0277 | — | — |
| Neural Network | 0.1781 | — | — |

> **Best model: SVR (R² = 0.9985)**  
> Tree-based and kernel models outperform deep learning models on this tabular time-series task, consistent with findings in financial ML literature.

### Key Findings

| Finding | Detail |
|---------|--------|
| **Best Model** | SVR — R² = 0.9985 (Exp 3) |
| **Top SHAP Feature** | `Return_Std7` (SHAP = 1.2604) |
| **Tweet Sentiment Rank** | 7th (via `Tweet_Sentiment_Lag3`) |
| **Price vs Tweet Correlation** | r = 0.075 (price), r = 0.195 (return) |
| **Price vs News Correlation** | r = 0.057 (price), r = 0.026 (return) |
| **High Volatility Events** | 43 events above threshold (4.77%) |
| **Sentiment-Aligned Events** | 5 of top 10 aligned with negative sentiment |

### Why LSTM/NN Underperform

Deep learning models require substantially larger datasets to generalise effectively. With only 429 rows (Exp 3) and 755–858 rows (Exp 1/2), the sequence models overfit and lack sufficient temporal patterns. Tree-based models handle small tabular datasets significantly better, which aligns with established literature on financial time series.

---

## References

```
[1] P. Giudici, "Explainable artificial intelligence methods for financial time series,"
    Physica A, vol. 641, 2024.

[2] T. L. Huynh, "Investor sentiment and cryptocurrency market dynamics,"
    IEEE Access, vol. 10, pp. 118451–118463, 2022.

[3] A. Kumar, R. K. Sharma, and S. Singh, "Social media sentiment and financial market
    volatility: A deep learning approach," IEEE Access, vol. 11, pp. 74231–74245, 2023.

[4] S. Nasekin and W. Chen, "Cryptocurrency volatility forecasting using machine learning
    techniques," IEEE TNNLS, vol. 32, no. 11, pp. 5105–5116, 2021.

[5] S. Corbet et al., "Cryptocurrencies as a financial asset: A systematic analysis,"
    International Review of Financial Analysis, vol. 62, pp. 182–199, 2019.

[6] Y. Peng et al., "Deep learning for cryptocurrency price prediction,"
    Expert Systems with Applications, vol. 141, 2020.

[7] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions,"
    NeurIPS, vol. 30, 2017.

[8] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks,"
    NeurIPS, vol. 33, 2020.

[9] O. Izacard and E. Grave, "Leveraging passage retrieval with generative models for
    open domain question answering," arXiv:2007.01282, 2020.

[10] A. Kraaijeveld and J. De Smedt, "The predictive power of public Twitter sentiment
     for forecasting cryptocurrency prices," JIMF, vol. 65, 2020.
```

---

<div align="center">

**COM748 Masters Research Project**  
Kaniz Fatema Roxy · B01036656 · Supervisor: Dr Nasir Iqbal

</div>

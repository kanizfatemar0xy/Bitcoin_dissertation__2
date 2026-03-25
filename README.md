# Bitcoin Volatility Prediction
## Using News & Social Media Sentiment with Explainable AI and Retrieval-Augmented Generation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/DL-TensorFlow-red?style=flat-square&logo=tensorflow" />
  <img src="https://img.shields.io/badge/XAI-SHAP-green?style=flat-square" />
  <img src="https://img.shields.io/badge/NLP-TextBlob-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square" />
</p>

> **COM748 Masters Research Project**
> Student: Kaniz Fatema Roxy &nbsp;|&nbsp; Supervisor: Dr Nasir Iqbal

---

## What This Project Does

Bitcoin is one of the most volatile financial assets in the world. This project builds an end-to-end research pipeline that:

- **Predicts** Bitcoin's 7-day rolling volatility using machine learning and deep learning models
- **Explains** which features drive those predictions using SHAP (Shapley Additive Explanations)
- **Contextualizes** high-volatility events by retrieving relevant news headlines and tweets and generating narrative explanations (RAG)

Unlike most existing studies that treat models as black boxes, this project integrates **explainability** at every level — making it possible to understand not just *what* the model predicts, but *why*.

---

## Table of Contents

- [Research Objectives](#research-objectives)
- [What Was Done](#what-was-done)
- [What Was Not Done and Why](#what-was-not-done-and-why)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Pipeline Overview](#pipeline-overview)
- [Feature Engineering](#feature-engineering)
- [Model Results](#model-results)
- [SHAP — Explainable AI](#shap--explainable-ai)
- [RAG — High Volatility Events](#rag--high-volatility-events)
- [Similarity Analysis](#similarity-analysis)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Installation and Usage](#installation-and-usage)

---

## Research Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Investigate the relationship between Bitcoin price volatility and sentiment from social media and news data | Done |
| 2 | Research how sentiment indicators trigger rapid volatility rises in the Bitcoin market | Done |
| 3 | Evaluate how well Explainable AI (SHAP) identifies the most significant features impacting volatility predictions | Done |
| 4 | Determine if Retrieval-Augmented Generation provides helpful narrative explanations consistent with quantitative volatility analysis | Done |

---

## What Was Done

### 1. Data Collection and Cleaning

Three publicly available datasets were collected from Kaggle and Yahoo Finance:

- **Bitcoin Historical Price Data** — 1,512 daily rows from 2021 to 2025 including Open, High, Low, Close, and Volume
- **Bitcoin Twitter Sentiment** — 11,295 individual records aggregated to 887 daily rows from 2021 to 2024, each with a sentiment score from -1 (very negative) to +1 (very positive)
- **Bitcoin News Headlines** — 785 daily rows from 2021 to 2023, where each row contains 10 Bitcoin news headlines processed with TextBlob to generate daily average sentiment scores

All three datasets were cleaned, filtered to 2021 onwards, and standardized to daily frequency before any analysis.

---

### 2. Similarity Analysis

A combined similarity plot was created showing Bitcoin price, Twitter sentiment, and News sentiment on a single timeline. Pearson correlation was calculated to quantify relationships:

| Comparison | Pearson r (vs Price) | Pearson r (vs Return) |
|-----------|---------------------|----------------------|
| Price vs Twitter Sentiment | 0.075 | **0.195** |
| Price vs News Sentiment | 0.057 | 0.026 |

The key finding was that **Twitter sentiment has a meaningfully stronger relationship with daily Bitcoin returns** (r = 0.195) than news sentiment does (r = 0.026). This confirms social media as the more responsive sentiment signal.

---

### 3. Feature Engineering and Master Dataset

All three datasets were merged on the Date column using an inner join, producing a master dataset of **429 rows and 43 columns**. Additional features were engineered:

- **Lag features** — price, returns, and sentiment values from 1, 2, 3, and 7 days prior (captures delayed effects)
- **Rolling features** — 7-day and 30-day moving averages for both price and sentiment
- **Volatility** — 30-day rolling volatility as an additional predictor
- **Price range** — daily High minus Low as a percentage of Close
- **Target variable** — `Volatility_7d`, the 7-day rolling standard deviation of daily returns

---

### 4. Machine Learning Models

Four ML models were trained on an **80/20 time-series aware split** (343 train, 86 test) with **MinMaxScaler** normalization. Each model was trained with default parameters and then fine-tuned using **GridSearchCV** with 3-fold cross-validation:

| Model | RMSE | MAE | R² | Best Parameters |
|-------|------|-----|----|-----------------|
| **SVR (Best)** | **0.0381** | **0.0333** | **0.9985** | kernel=linear, C=1.0, epsilon=0.05 |
| Gradient Boosting | 0.0754 | 0.0407 | 0.9941 | lr=0.1, depth=3, n=200 |
| XGBoost | 0.0954 | 0.0582 | 0.9906 | lr=0.1, depth=3, n=200 |
| Random Forest | 0.1194 | 0.0648 | 0.9853 | depth=None, split=2, n=200 |

SVR with a linear kernel achieved the best performance with R² = 0.9985.

---

### 5. Deep Learning Models

Two deep learning architectures were trained using EarlyStopping and ReduceLROnPlateau callbacks:

- **LSTM** — two stacked LSTM layers with Dropout and BatchNormalization, using a 7-day sequence lookback window
- **Neural Network** — four fully connected Dense layers with Dropout and BatchNormalization

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| LSTM (Tuned) | 1.0351 | 0.8484 | -0.0277 |
| Neural Network (Tuned) | 0.8912 | 0.7062 | 0.1781 |

Both models significantly underperformed. The reason is explained in detail below.

---

### 6. SHAP — Explainable AI

SHAP values were calculated for all four ML models:
- `TreeExplainer` was used for Random Forest, XGBoost, and Gradient Boosting (fast and exact)
- `KernelExplainer` with 50 background samples was used for SVR (slower, sample-based)

Results were combined into a mean importance ranking across all models, producing bar plots, a beeswarm plot for XGBoost, and a combined cross-model comparison chart.

**Top 10 Features (Mean |SHAP| across all models):**

| Rank | Feature | Mean SHAP | Category |
|------|---------|-----------|----------|
| 1 | Return_Std7 | 1.2604 | Price |
| 2 | Close_MA30 | 0.0310 | Price |
| 3 | Close_Lag7 | 0.0139 | Price |
| 4 | Volatility_30d | 0.0137 | Price |
| 5 | Volume_MA7 | 0.0076 | Price |
| 6 | Close | 0.0069 | Price |
| 7 | **Tweet_Sentiment_Lag3** | **0.0059** | Social Media |
| 8 | Close_MA7 | 0.0039 | Price |
| 9 | Price_Range_Pct | 0.0035 | Price |
| 10 | Return_Lag3 | 0.0026 | Price |

`Return_Std7` dominates all other features by approximately **40x**. `Tweet_Sentiment_Lag3` at rank 7 reveals that Twitter sentiment from **3 days prior** has a measurable and consistent influence on Bitcoin volatility across all four models.

---

### 7. RAG — Retrieval-Augmented Generation

High-volatility events were identified using the **90th percentile threshold (4.77%)**, yielding **43 events**. For each of the top 5 events, a structured pipeline was run:

1. **Detection** — event identified as exceeding the volatility threshold
2. **Retrieval** — news articles and tweets from a ±3-day window are fetched
3. **Generation** — a narrative is generated combining SHAP drivers, sentiment context, and event classification (Bull Spike or Bear Crash)

**Top 5 Events:**

| Date | Price | Return | Volatility | Type | Sentiment Aligned |
|------|-------|--------|-----------|------|------------------|
| 2022-11-10 | $17,587 | +10.74% | 8.46% | Bull Spike | No |
| 2022-11-14 | $16,618 | +1.62% | 8.07% | Bull Spike | Yes |
| 2022-06-19 | $20,553 | +8.07% | 8.04% | Bull Spike | Yes |
| 2022-11-11 | $17,034 | -3.14% | 7.99% | Bear Crash | Yes |
| 2022-11-12 | $16,799 | -1.38% | 7.89% | Bear Crash | Yes |

The November 2022 cluster corresponds to the **FTX exchange collapse**. 5 out of 10 top events showed sentiment aligned with price direction.

---

## What Was Not Done and Why

### Deep Learning Did Not Achieve Good Results

This was expected, not a bug. LSTM and Neural Network models require a large number of training samples — typically 1,000 or more — to learn meaningful temporal patterns. Our master dataset contained only **343 training rows**, which is far too small. This limitation arose directly from the narrow overlap window between all three datasets (Dec 2021 to Feb 2023).

The result is actually a valid research finding: for small financial time-series datasets, traditional ML models such as SVR and Gradient Boosting significantly outperform deep learning architectures.

---

### Separate Model Training Per Dataset Was Not Possible

The three datasets could not be used to train three separate models because the **target variable** (`Volatility_7d`) only exists in the price dataset. The tweets file and news file contain only input features — they have no output variable to predict. Supervised learning requires both features and a target in the same table, so merging all three files by date was the only valid approach.

---

### FinBERT Was Not Used

TextBlob was used for news headline sentiment scoring. It is a general-purpose tool not trained on financial language, which explains the narrow sentiment range (-0.15 to +0.28) in the news data. A domain-specific model like **FinBERT** would produce more nuanced and accurate scores but was outside the scope of this project.

---

### No Real-Time Data Integration

The system uses historical data only. Live Bitcoin price APIs, real-time Twitter scraping, and live news feeds were not integrated. This is a clear direction for future work.

---

### RAG Used Rule-Based Generation Instead of an LLM

The RAG pipeline retrieves relevant context (news and tweets) and fills structured narrative templates with quantitative values. A full LLM-powered RAG system would call a model like GPT-4 to generate dynamic, contextually rich explanations. This was not implemented due to API cost and scope constraints.

---

## Project Structure

```
bitcoin_volatility_project/
│
├── data/
│   ├── 1_price_data.csv                 # Bitcoin price (2021-2025)
│   ├── 2_tweets_data.csv                # Twitter sentiment daily (2021-2024)
│   ├── 3_news_data.csv                  # News sentiment daily (2021-2023)
│   ├── master_dataset.csv               # Merged — 429 rows, 43 columns
│   ├── ml_results.csv                   # ML model evaluation metrics
│   ├── all_models_results.csv           # All 7 models final comparison
│   └── shap_feature_importance.csv      # Mean SHAP per feature
│
├── models/
│   ├── scaler.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── svr_model.pkl
│   ├── lstm_model.keras
│   └── nn_model.keras
│
├── plots/
│   ├── individual/                      # plot1_price.png, plot2_tweets.png, plot3_news.png
│   ├── similarity/                      # plot_price_vs_tweets_vs_news.png
│   ├── ml_results/                      # Actual vs predicted + comparison chart
│   ├── shap/                            # Bar plots + beeswarm per model
│   └── rag/                             # Volatility events + event window plots
│
├── rag_results/
│   ├── rag_narratives.txt
│   ├── rag_narratives.json
│   └── rag_summary_table.csv
│
└── scripts/
    ├── clean_and_aggregate.py
    ├── similarity_combined.py
    ├── generate_plots_simple.py
    ├── create_master_dataset.py
    ├── train_ml_models.py
    ├── train_dl_models.py
    ├── shap_analysis.py
    └── rag_pipeline.py
```

---

## Datasets

| Dataset | Link |
|---------|------|
| Bitcoin Historical Price (2014-2025) | [Kaggle](https://www.kaggle.com/datasets/eldintarofarrandi/bitcoin-historical-data-2014-2025-yahoo-finance) |
| Bitcoin Twitter Sentiment | [Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/bitcoin-tweets) |
| Bitcoin News Headlines (BTC.csv) | [Kaggle](https://www.kaggle.com/datasets/aaroncbastian/crypto-news-headlines-and-market-prices-by-date) |

Place all downloaded files in the `data/` folder before running any scripts.

---

## Pipeline Overview

```
Raw Data (3 CSV files)
        |
        v
Data Cleaning             scripts/clean_and_aggregate.py
        |
        v
Similarity Analysis       scripts/similarity_combined.py
        |
        v
EDA Plots                 scripts/generate_plots_simple.py
        |
        v
Feature Engineering       scripts/create_master_dataset.py
+ Merge to Master CSV
        |
        v
ML Training               scripts/train_ml_models.py
RF + XGB + GB + SVR       GridSearchCV fine-tuning
        |
        v
Deep Learning             scripts/train_dl_models.py
LSTM + NN                 EarlyStopping fine-tuning
        |
        v
SHAP Analysis             scripts/shap_analysis.py
All 4 ML models
        |
        v
RAG Pipeline              scripts/rag_pipeline.py
Event detection +
retrieval + narratives
```

---

## Feature Engineering

| Group | Features | Count |
|-------|----------|-------|
| Price | Close, Open, High, Low, Volume, Daily_Return | 6 |
| Volatility | Volatility_30d + Volatility_7d (TARGET) | 2 |
| Tweet | Tweet_Sentiment, Tweet_Count, Positive, Negative, Neutral | 5 |
| News | News_Sentiment, News_Count, Positive, Negative, Neutral | 5 |
| Lag 1,2,3,7 days | Close_Lag, Return_Lag, Tweet_Sentiment_Lag, News_Sentiment_Lag | 12 |
| Rolling | Close_MA7, Close_MA30, Return_Std7, Tweet_Sent_MA7, News_Sent_MA7 | 5 |
| Price Range | Price_Range, Price_Range_Pct | 2 |
| **Total** | | **41 features** |

---

## Model Results

**Split:** 80/20 time-series aware — 343 train, 86 test
**Scaler:** MinMaxScaler fitted on training data only

| Rank | Model | RMSE | MAE | R2 |
|------|-------|------|-----|-----|
| 1 | SVR | 0.0381 | 0.0333 | 0.9985 |
| 2 | Gradient Boosting | 0.0754 | 0.0407 | 0.9941 |
| 3 | XGBoost | 0.0954 | 0.0582 | 0.9906 |
| 4 | Random Forest | 0.1194 | 0.0648 | 0.9853 |
| 5 | Neural Network | 0.8912 | 0.7062 | 0.1781 |
| 6 | LSTM | 1.0351 | 0.8484 | -0.0277 |

---

## SHAP — Explainable AI

SHAP was applied using `TreeExplainer` for tree models and `KernelExplainer` for SVR. A combined mean importance ranking was produced across all four models.

Key findings:
- `Return_Std7` is the dominant predictor with SHAP value 40x higher than the next feature
- `Tweet_Sentiment_Lag3` ranks 7th — confirming a 3-day delayed sentiment effect on volatility
- Price-based features dominate the top 6 ranks, confirming that market behavior is primarily self-referential

---

## RAG — High Volatility Events

Each narrative covers:

- Date, price, return, and volatility level
- Top SHAP features driving the prediction
- Retrieved news and tweet context from ±3 days
- Plain-language explanation classifying the event as Bull Spike or Bear Crash
- Sentiment alignment check — whether sentiment agreed with price direction

---

## Similarity Analysis

| Metric | Twitter | News |
|--------|---------|------|
| Pearson r vs Price | 0.075 | 0.057 |
| Pearson r vs Daily Return | 0.195 | 0.026 |
| Coverage (days) | 887 | 785 |

Twitter sentiment is a stronger predictor of daily returns than news sentiment, suggesting social media captures and reflects market mood more responsively than formal news outlets.

---

## Key Findings

| Finding | Detail |
|---------|--------|
| Best Model | SVR — R2 = 0.9985 with linear kernel |
| ML vs Deep Learning | ML significantly outperforms DL due to limited training data (343 rows) |
| Top Feature | Return_Std7 — 40x more important than any other single feature |
| Sentiment Impact | Tweet_Sentiment_Lag3 is the top sentiment feature — 3-day delayed market reaction |
| Twitter vs News | Twitter r=0.195 vs News r=0.026 for daily return correlation |
| Sentiment Alignment | 50% of top 10 high-volatility events showed sentiment aligned with price direction |
| Biggest Event | FTX collapse November 2022 — volatility cluster of 7.89% to 8.46% |

---

## Limitations

| Limitation | Explanation |
|-----------|-------------|
| Small data overlap | Only 429 rows in master dataset due to limited intersection of three sources |
| News data ends 2023 | BTC.csv ends February 2023 — no news features for 2023 to 2025 |
| TextBlob sentiment | General-purpose tool; FinBERT would be more accurate for financial text |
| No real-time feeds | Historical data only — live prediction not supported |
| Rule-based RAG | Narrative templates used instead of a live LLM |
| Separate file training not possible | Target variable exists only in price data — merge was required |

---

## Installation and Usage

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost
pip install tensorflow shap joblib textblob
```

### Run All Scripts in Order

```bash
cd scripts

py clean_and_aggregate.py       # Clean all 3 datasets
py similarity_combined.py       # Combined similarity plot
py generate_plots_simple.py     # Individual EDA plots
py create_master_dataset.py     # Merge + feature engineering
py train_ml_models.py           # Train RF, XGB, GB, SVR
py train_dl_models.py           # Train LSTM and NN
py shap_analysis.py             # SHAP feature importance
py rag_pipeline.py              # RAG event narratives
```

> Update the `BASE` path variable at the top of each script to match your local directory before running.

---

<p align="center">
<em>COM748 Masters Research Project — Bitcoin Volatility Using News and Social Media with Explainable AI and RAG</em>
</p>

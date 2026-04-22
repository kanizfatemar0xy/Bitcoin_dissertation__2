#  Bitcoin Volatility Prediction
### Using News & Social Media Sentiment with Explainable AI (SHAP) and Retrieval-Augmented Generation (RAG)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-XAI-00C853?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Narratives-7B1FA2?style=for-the-badge)

<br/>

> **Masters Research Project — COM748**
> *Kaniz Fatema Roxy (B01036656) · Supervisor: Dr Nasir Iqbal*

<br/>

**[ View Results](#-results--model-performance) · [ Quick Start](#-installation--quick-start) · [ Architecture](#-system-architecture) · [📁 Project Structure](#-project-structure)**

</div>

---

##  What This Project Does

Traditional Bitcoin volatility models are **black boxes** — they predict *when* volatility happens but can't explain *why*. This project solves that.

| Challenge | Our Solution |
|-----------|-------------|
| Black-box AI models |  SHAP Explainable AI — shows which features drive predictions |
| No context for volatility spikes |  RAG Pipeline — generates human-readable narrative explanations |
| Single-source prediction |  3 Experiments — Price only, +News, +Tweets, +All combined |
| Static analysis only |  Live Flask Dashboard — real-time predictions using yfinance + RSS |

---

##  Research Objectives

1. Investigate the relationship between Bitcoin price volatility and social media / news sentiment
2. Measure how sentiment indicators trigger rapid volatility spikes
3. Evaluate SHAP's ability to identify the most influential features
4. Determine whether RAG provides meaningful narrative explanations aligned with quantitative analysis

---

##  System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                         DATA LAYER                                   ║
║    Yahoo Finance (Price)   Kaggle Tweets   Kaggle News               ║
╚══════════════════════════════╦═══════════════════════════════════════╝
                               ║
                               ▼
╔═══════════════════════════════════════════════════════════════════════╗
║                    FEATURE ENGINEERING                                ║
║  Daily Returns · 7d/30d Rolling Volatility · MA7 · MA30               ║
║  Lag Features (1,2,3,7 days) · TextBlob Sentiment · Price Range       ║
╚════════════╦══════════════════╦═══════════════════════════════════════╝
             ║                  ║                  ║
             ▼                  ▼                  ▼
       ┌──────────┐      ┌──────────┐      ┌──────────────┐
       │  Exp 1   │      │  Exp 2   │      │   Exp 3      │
       │Price+News│      │Price+Twt │      │  All Three   │
       │ ~755 rows│      │ ~858 rows│      │  429 rows    │
       └────┬─────┘      └────┬─────┘      └──────┬───────┘
            └─────────────────┴──────────────────┘
                               ║
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║                        6 ML MODELS                                   ║
║  Random Forest · XGBoost · Gradient Boosting · SVR · LSTM · NN       ║
╚════════════╦═════════════════════════════════════╦═══════════════════╝
             ║                                     ║
             ▼                                     ▼
    ┌─────────────────┐                  ┌──────────────────────┐
    │   SHAP (XAI)    │                  │    RAG PIPELINE      │
    │ TreeExplainer   │                  │ Event Detection      │
    │ KernelExplainer │                  │ TF-IDF Retrieval     │
    │ Feature Rankings│                  │ Narrative Generation │
    └────────┬────────┘                  └──────────┬───────────┘
             └─────────────────┬─────────────────────┘
                               ▼
            ╔══════════════════════════════════════╗
            ║       FLASK WEB DASHBOARD            ║
            ║  Live & Historical Prediction        ║
            ║  SHAP Plots · RAG Summary            ║
            ║  RMSE / MAE / R² Inline Display      ║
            ╚══════════════════════════════════════╝
```

---

##  Datasets

| # | Dataset | Source | Raw | Filtered | Date Range |
|---|---------|--------|-----|---------|-----------|
| 1 | **Bitcoin Price** | Yahoo Finance via Kaggle | 3,810 rows | **1,512 rows** | Jan 2021 → Feb 2025 |
| 2 | **Bitcoin Tweets** | Kaggle | 11,295 records | **887 rows** (daily agg.) | Nov 2021 → Sep 2024 |
| 3 | **Bitcoin News** | Kaggle (BTC.csv) | 1,826 rows | **785 rows** | Jan 2021 → Feb 2023 |

**Target Variable:** `Volatility_7d` — 7-day rolling standard deviation of daily returns

**Features per dataset:** Price columns + Sentiment scores + Lag features (1,2,3,7 days) + Rolling averages (MA7, MA30, Return_Std7)

---

##  Experiments

Three controlled experiments with progressively richer sentiment data:

| | **Exp 1** | **Exp 2** | **Exp 3** |
|---|-----------|-----------|-----------|
| **Data** | Price + News | Price + Tweets | Price + Tweets + News |
| **Rows** | ~755 | ~858 | 429 |
| **Features** | ~47 | ~47 | 43 |
| **Train / Test** | ~604 / ~151 | ~686 / ~172 | 343 / 86 |
| **Date Range** | Jan 2021 – Feb 2023 | Nov 2021 – Sep 2024 | Dec 2021 – Feb 2023 |
| **Coverage** | ~26 months | ~34 months | ~14 months |

> All experiments use **80/20 chronological split** — no data leakage.

---

##  Models Trained

| Model | Type | Notes |
|-------|------|-------|
| **Random Forest** | Ensemble (Tree) | 200 estimators |
| **XGBoost** | Gradient Boosting | lr=0.05, depth=6 |
| **Gradient Boosting** | Ensemble (Tree) | lr=0.05, depth=4 |
| **SVR** | Kernel Method | RBF kernel, C=10 |
| **LSTM** | Deep Learning | 64→32 units, timesteps=10 |
| **Neural Network** | Deep Learning | 128→64→32 units |

**Total models trained: 6 models × 3 experiments = 18 models**

---

##  Results & Model Performance

###  Experiment 1 — Price + News (~755 rows)

| Model | R² | RMSE | MAE | Verdict |
|-------|-----|------|-----|---------|
| Random Forest | 0.9942 | 0.1152 | 0.0520 |  Excellent |
| XGBoost | 0.9973 | 0.0790 | 0.0392 |  Excellent |
| **Gradient Boosting** | **0.9987** | **0.0537** | **0.0247** |  Best in Exp 1 |
| SVR | 0.8558 | 0.5753 | 0.3984 |  Moderate |
| LSTM | −0.1584 | 1.6799 | 1.4597 |  Poor |
| Neural Network | 0.9685 | 0.2688 | 0.2083 |  Good |

###  Experiment 2 — Price + Tweets (~858 rows)

| Model | R² | RMSE | MAE | Verdict |
|-------|-----|------|-----|---------|
| **Random Forest** | **0.8898** | **0.4059** | **0.2143** |  Best in Exp 2 |
| XGBoost | 0.8704 | 0.4403 | 0.2589 |  Good |
| Gradient Boosting | 0.8720 | 0.4374 | 0.2549 |  Good |
| SVR | 0.2656 | 1.0480 | 0.8156 |  Poor |
| LSTM | −0.1930 | 1.3320 | 0.9488 |  Poor |
| Neural Network | 0.8271 | 0.5085 | 0.3832 |  Moderate |

###  Experiment 3 — Price + Tweets + News (429 rows)

| Model | R² | RMSE | MAE | Verdict |
|-------|-----|------|-----|---------|
| Random Forest | 0.9855 | 0.1182 | 0.0662 |  Excellent |
| XGBoost | 0.9841 | 0.1241 | 0.0696 |  Excellent |
| **Gradient Boosting** | **0.9918** | **0.0891** | **0.0536** |  Best in Exp 3 |
| SVR | 0.8702 | 0.3541 | 0.2743 |  Moderate |
| LSTM | 0.3100 | 0.8585 | 0.7243 |  Poor |
| Neural Network | 0.6969 | 0.5412 | 0.4525 |  Moderate |

---

###  Overall Best Models (Cross-Experiment)

| Rank | Model | Best Experiment | R² | RMSE | MAE |
|------|-------|----------------|-----|------|-----|
|  1st | **Gradient Boosting** | Exp 1 (Price+News) | **0.9987** | **0.0537** | **0.0247** |
|  2nd | XGBoost | Exp 1 (Price+News) | 0.9973 | 0.0790 | 0.0392 |
|  3rd | Random Forest | Exp 1 (Price+News) | 0.9942 | 0.1152 | 0.0520 |

> **Key Insight:** Experiment 1 (Price + News) consistently outperforms Exp 2 and Exp 3 for tree-based models, suggesting **news sentiment is a stronger predictor of volatility than tweet sentiment** for this dataset.

---

###  Why LSTM & SVR Underperform

**LSTM:** Deep learning sequence models require significantly larger datasets to generalise. With 86–172 test samples, the model lacks sufficient temporal patterns and overfits on training data. This is consistent with findings in financial ML literature where tree-based models outperform deep learning on small tabular datasets.

**SVR (Exp 2):** SVR struggled specifically with the tweet-only sentiment experiment (R² = 0.27), likely due to the high day-to-day noise in social media sentiment signals, which makes the decision boundary harder to separate with a kernel function.

---

##  Explainable AI — SHAP Results

SHAP values computed for all 6 models × 3 experiments using:

| Model | Explainer Used |
|-------|---------------|
| Random Forest, XGBoost, Gradient Boosting | `TreeExplainer` (fast, exact) |
| SVR | `KernelExplainer` |
| Neural Network | `KernelExplainer` |
| LSTM | `KernelExplainer` (flattened sequences, averaged over timesteps) |

**Key SHAP Finding (Exp 3 — All Features):**

```
Top Features by Mean |SHAP Value|:
  1. Return_Std7              (dominant — rolling volatility signal)
  2. Volatility_7d_Lag1       (yesterday's volatility)
  3. Close_Lag1               (yesterday's price)
  4. Daily_Return_Lag1
  5. MA7
  6. Volatility_7d_Lag3
  7. Tweet_Sentiment_Lag3     ← sentiment enters top 10
  8. News_Avg_Sentiment_Lag1  ← news sentiment also present
```

> Sentiment features consistently rank in the **top 10** but below price-derived features, confirming that while sentiment influences volatility, **historical price volatility is the dominant predictor**.

---

##  RAG — Retrieval-Augmented Generation

The RAG pipeline automatically identifies and explains the **top 10 highest-volatility events** per experiment.

**How it works:**

```
Step 1 → Detect events where Volatility_7d > mean + 1.5σ
Step 2 → Retrieve surrounding ±3 day context using TF-IDF similarity
Step 3 → Generate structured narrative linking price, sentiment & market events
Step 4 → Save as .txt, .json, .csv + visualisation plots
```

**Notable events captured:**

| Date | Volatility Level | Market Context |
|------|----------------|----------------|
| 2022-05-xx | EXTREME | Terra/LUNA collapse — crypto market crash |
| 2022-11-xx | EXTREME | FTX collapse — major crypto crisis |
| 2021-05-xx | HIGH | China crypto mining ban |
| 2022-03-xx | HIGH | Russia-Ukraine war — market uncertainty |
| 2021-11-xx | HIGH | Bitcoin new ATH near $69K |

**Finding:** 5 of the top 10 high-volatility events aligned with **negative news sentiment**, supporting the hypothesis that negative sentiment amplifies volatility.

---

##  Web Dashboard

A live Flask dashboard for interactive prediction:

```
http://127.0.0.1:5000
```

**Features:**

| Feature | Description |
|---------|-------------|
|  Experiment Selector | Switch between Exp 1, 2, or 3 |
|  Any Date | Historical OR live — works for any date |
|  Model Selector | All 6 models available |
|  Live Mode | Today → yfinance + Google News RSS + CoinDesk RSS |
|  Historical Mode | Exact CSV row, or nearest-neighbour fallback |
|  Prediction | Next-day volatility forecast |
|  RMSE/MAE/R² | Model accuracy shown inline on results page |
|  RAG Summary | Nearest event narrative from RAG database |
|  SHAP Plot | Feature impact chart per selected model |

---

##  Project Structure

```
Bitcoin/
│
├── 📁 data/
│   ├── 1_price_data.csv              # Raw Bitcoin price (2021–2025)
│   ├── 2_tweets_data.csv             # Raw tweet sentiment (daily)
│   ├── 3_news_data.csv               # Raw news sentiment (daily)
│   ├── master_price_news.csv         # Exp 1 — ~755 rows, ~47 features
│   ├── master_price_tweets.csv       # Exp 2 — ~858 rows, ~47 features
│   └── master_dataset.csv            # Exp 3 — 429 rows, 43 features
│
├── 📁 models/
│   ├── exp1/                         # 6 trained models + scalers + results
│   ├── exp2/                         # (same structure)
│   └── exp3/                         # (same structure)
│         ├── random_forest_model.pkl
│         ├── xgboost_model.pkl
│         ├── gradient_boosting_model.pkl
│         ├── svr_model.pkl
│         ├── lstm_model.keras
│         ├── nn_model.keras
│         ├── scaler.pkl / y_scaler.pkl
│         └── results.csv
│
├── 📁 plots/
│   ├── individual/                   # Price, Tweet, News line plots
│   ├── similarity/                   # Normalized comparison plots
│   └── experiments/
│       ├── exp1_price_news/
│       ├── exp2_price_tweets/
│       └── exp3_all/
│
├── 📁 shap/
│   ├── exp1/                         # Bar + Beeswarm + Comparison + CSV
│   ├── exp2/
│   └── exp3/
│
├── 📁 rag/
│   ├── exp1/                         # narratives.txt/.json, summary.csv, plots
│   ├── exp2/
│   └── exp3/
│
├── 📁 results/
│   ├── exp1_results.csv              # R², RMSE, MAE — Exp 1
│   ├── exp2_results.csv              # R², RMSE, MAE — Exp 2
│   ├── exp3_results.csv              # R², RMSE, MAE — Exp 3
│   ├── all_experiments_comparison.csv
│   └── plots/
│       ├── exp1/2/3_metrics_comparison.png
│       ├── all_r2/rmse/mae_comparison.png
│       └── actual_vs_predicted/      # 18 plots (6 models × 3 experiments)
│
├── 📁 scripts/                       # Run in order ↓
│   ├── create_master_datasets.py     # Step 1 — build datasets
│   ├── generate_all_plots.py         # Step 2 — EDA plots
│   ├── train_all_experiments.py      # Step 3 — train 18 models
│   ├── shap_analysis.py             # Step 4 — SHAP explanations
│   ├── rag_pipeline.py              # Step 5 — RAG narratives
│   └── evaluate_all_experiments.py  # Step 6 — RMSE/MAE/R² evaluation
│
├── 📁 templates/
│   ├── index.html                    # Dashboard home
│   └── results.html                  # Prediction results
│
├── app.py                            # 🌐 Flask web application
└── README.md                         # 📖 This file
```

---

##  Installation & Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost tensorflow shap flask yfinance textblob
```

### 2. Place raw data files in `data/`

| File to rename | Kaggle Dataset |
|---------------|----------------|
| `1_price_data.csv` | [Bitcoin Historical Data 2014–2025](https://www.kaggle.com/datasets/eldintarofarrandi/bitcoin-historical-data-2014-2025-yahoo-finance) |
| `2_tweets_data.csv` | [Bitcoin Tweets](https://www.kaggle.com/datasets/sujaykapadnis/bitcoin-tweets) |
| `3_news_data.csv` | [Bitcoin News Dataset](https://www.kaggle.com/datasets/ashirwadsangwan/bitcoinnews-dataset) |

### 3. Run the pipeline — in order

```bash
# Step 1 — Create master datasets (~2 min)
py scripts/create_master_datasets.py

# Step 2 — Generate EDA plots (~1 min)
py scripts/generate_all_plots.py

# Step 3 — Train all 18 models (~30–45 min)
py scripts/train_all_experiments.py

# Step 4 — SHAP analysis (~20–30 min)
py scripts/shap_analysis.py

# Step 5 — RAG pipeline (~2 min)
py scripts/rag_pipeline.py

# Step 6 — Evaluate RMSE/MAE/R² (~5–10 min)
py scripts/evaluate_all_experiments.py

# Step 7 — Launch dashboard
py app.py
```

**Then open:** `http://127.0.0.1:5000`

---

##  References

| # | Citation |
|---|---------|
| [1] | P. Giudici, *"Explainable AI methods for financial time series,"* Physica A, vol. 641, 2024 |
| [2] | T. L. Huynh, *"Investor sentiment and cryptocurrency market dynamics,"* IEEE Access, vol. 10, 2022 |
| [3] | A. Kumar et al., *"Social media sentiment and financial market volatility,"* IEEE Access, vol. 11, 2023 |
| [4] | S. Nasekin & W. Chen, *"Cryptocurrency volatility forecasting using ML,"* IEEE TNNLS, vol. 32, 2021 |
| [5] | S. Corbet et al., *"Cryptocurrencies as a financial asset,"* IRFA, vol. 62, 2019 |
| [6] | Y. Peng et al., *"Deep learning for cryptocurrency price prediction,"* Expert Systems, vol. 141, 2020 |
| [7] | S. M. Lundberg & S.-I. Lee, *"A unified approach to interpreting model predictions,"* NeurIPS, 2017 |
| [8] | P. Lewis et al., *"Retrieval-augmented generation for NLP tasks,"* NeurIPS, 2020 |
| [9] | O. Izacard & E. Grave, *"Leveraging passage retrieval with generative models,"* arXiv:2007.01282, 2020 |
| [10] | A. Kraaijeveld & J. De Smedt, *"Predictive power of Twitter for cryptocurrency prices,"* JIMF, 2020 |

---

<div align="center">

---

**COM748 Masters Research Project**

*Kaniz Fatema Roxy · B01036656 · Supervisor: Dr Nasir Iqbal*

---

</div>

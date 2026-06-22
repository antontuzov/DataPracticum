# Data Science Practicum — Master Course Structure

> **Role**: Senior Principal Data Scientist & Curriculum Architect
> **Target Domains**: Fintech (Revolut) · Big Tech & Search (Yandex) · Digital Marketing
> **Total Notebooks**: 52 fully functional, self-contained Jupyter Notebooks
> **Tech Stack**: Python, Pandas, PyTorch, Scikit-learn, Statsmodels, Matplotlib, Seaborn, Plotly

---

## How to Use This Course

1. Browse the four tracks below.
2. Type the command shown (e.g. `Build Module 1`) to generate the complete notebook.
3. Every notebook is **self-contained** — synthetic data is generated internally, no external files needed.

---

## Track 1 — Data Science Foundations (10 Notebooks)

| # | Command | Title | Description |
|---|---------|-------|-------------|
| 1 | `Build Module 1` | **Optimized Pandas: Vectorization vs. Loops vs. Apply** | Benchmarks three approaches on a 1M-row financial dataset; demonstrates why vectorized code is 100× faster and teaches `.eval()`, `.query()`, and NumPy broadcasting patterns. |
| 2 | `Build Module 2` | **Advanced GroupBy & Pivot Tables for Financial Aggregation** | Builds multi-level pivot tables from simulated Revolut-style transaction logs; covers `agg`, `transform`, `rolling`, and `expanding` windows for daily/weekly/monthly summaries. |
| 3 | `Build Module 3` | **Time-Series Resampling: OHLC Data for Stocks & Crypto** | Generates synthetic minute-level crypto trades, resamples to OHLC candles, plots candlestick charts with Plotly, and computes Bollinger Bands and RSI indicators. |
| 4 | `Build Module 4` | **PyTorch Tensors vs. NumPy: A Practical Comparison** | Side-by-side benchmarks of tensor vs. array operations; covers GPU transfer, autograd basics, and when to choose each library for data-science workloads. |
| 5 | `Build Module 5` | **Building a Custom DataLoader for Large Datasets** | Creates a PyTorch `Dataset` and `DataLoader` from a synthetic CSV of 5M rows; demonstrates batching, shuffling, multi-worker loading, and memory-mapped file reading. |
| 6 | `Build Module 6` | **Statistical Tests for A/B Testing** | Implements t-tests, Mann-Whitney U, Chi-Squared, and bootstrap confidence intervals on simulated A/B experiment data; covers power analysis and minimum sample-size calculation. |
| 7 | `Build Module 7` | **Handling Imbalanced Data: SMOTE, ADASYN & Weighted Loss** | Compares oversampling (SMOTE, ADASYN), undersampling (Tomek Links), and class-weight approaches on a 1%-positive fraud dataset; evaluates with PR-AUC. |
| 8 | `Build Module 8` | **Hyperparameter Tuning: GridSearch vs. Random vs. Bayesian** | Tunes an XGBoost classifier on synthetic credit data using three strategies; compares wall-clock time vs. achieved AUC and introduces Optuna for Bayesian optimization. |
| 9 | `Build Module 9` | **Feature Selection: Mutual Information, LASSO & SHAP** | Ranks 50 engineered features using filter (mutual info), wrapper (RFE), and embedded (LASSO, SHAP importance) methods; shows how feature selection improves model latency. |
| 10 | `Build Module 10` | **End-to-End Scikit-learn Pipeline** | Assembles a production-grade pipeline with preprocessing (imputer → scaler → encoder), feature selection, and model; demonstrates cross-validation, persistence with `joblib`, and inference. |

---

## Track 2 — Revolut Fintech Specialization (14 Notebooks)

| # | Command | Title | Description |
|---|---------|-------|-------------|
| 11 | `Build Module 11` | **Rule-Based Fraud Heuristics: Velocity & Threshold Detection** | Engineers velocity-burst rules (e.g., 5+ ATM withdrawals in 10 min), geo-impossibility checks, and round-amount flags on synthetic transaction streams; measures precision/recall of each rule. |
| 12 | `Build Module 12` | **Anomaly Detection with Isolation Forest** | Trains an Isolation Forest on 30 transaction features; visualizes anomaly scores in 2-D with t-SNE and compares contamination parameter tuning against a labeled holdout set. |
| 13 | `Build Module 13` | **Logistic Regression for Fraud: An Interpretable Baseline** | Builds a logistic-regression fraud model; covers odds-ratio interpretation, coefficient analysis, threshold selection via Youden's J, and a business cost-matrix for false positives vs. false negatives. |
| 14 | `Build Module 14` | **Imbalanced Fraud Handling: Weighted Cross-Entropy in PyTorch** | Implements a custom `WeightedBCELoss` in PyTorch; trains a small feed-forward network on the imbalanced fraud dataset and compares PR-AUC against the logistic-regression baseline. |
| 15 | `Build Module 15` | **Autoencoders for Unsupervised Anomaly Detection** | Builds a vanilla autoencoder in PyTorch; trains on "normal" transactions only and flags high-reconstruction-error rows as anomalies; compares against Isolation Forest from Module 12. |
| 16 | `Build Module 16` | **Time-Series Forecasting: Daily Transaction Volume** | Uses Statsmodels SARIMAX and Facebook Prophet to forecast 90 days of daily transaction volume; evaluates with MAPE and visualizes prediction intervals. |
| 17 | `Build Module 17` | **LSTM for Predicting Daily Active Users (DAU)** | Builds a PyTorch LSTM with attention to forecast DAU 7 days ahead; covers sequence creation, teacher forcing, and gradient clipping for stable training. |
| 18 | `Build Module 18` | **Customer Churn Prediction: Logistic Regression & XGBoost** | Compares logistic regression, random forest, and XGBoost on a synthetic Revolut user-attrition dataset; uses SHAP to explain top churn drivers. |
| 19 | `Build Module 19` | **Graph Neural Networks (Simplified): Money-Mule Ring Detection** | Generates a synthetic transaction graph with `networkx`; detects circular money flows and applies a simplified GNN (message-passing with PyTorch) to flag mule accounts. |
| 20 | `Build Module 20` | **Real-Time Feature Engineering: Rolling Statistics** | Engineers rolling features (7-day MA, 30-day std, max-in-24h) over simulated streaming transactions; demonstrates efficient Pandas `.rolling()` and incremental update patterns. |
| 21 | `Build Module 21` | **A/B Testing for UI Changes: Conversion-Rate Analysis** | Simulates a 50/50 split test for a new "Send Money" button; runs two-proportion z-test, Bayesian A/B (Beta-Binomial), and computes minimum detectable effect size. |
| 22 | `Build Module 22` | **Credit Risk Scoring: GBM for Default Probability** | Trains a LightGBM / GradientBoosting model on synthetic loan-application data; calibrates probabilities with Platt scaling and builds a scorecard mapping. |
| 23 | `Build Module 23` | **SHAP Explainability for Fraud Decisions** | Computes SHAP values for the fraud model from Module 13; builds waterfall plots, force plots, and summary plots; demonstrates how to generate human-readable explanations for compliance. |
| 24 | `Build Module 24` | **Data Drift Detection: PSI & Model Degradation Monitoring** | Calculates Population Stability Index (PSI) across monthly data slices; builds a drift dashboard with Plotly that flags when retraining is needed. |

---

## Track 3 — Yandex Big Tech & Search / Recommendation (13 Notebooks)

| # | Command | Title | Description |
|---|---------|-------|-------------|
| 25 | `Build Module 25` | **Text Preprocessing: Tokenization, Stemming & Stopwords** | Builds a preprocessing pipeline for English and Russian text; covers regex cleaning, `nltk`/`pymorphy2` tokenization, lemmatization, and stopword removal on synthetic product descriptions. |
| 26 | `Build Module 26` | **TF-IDF & BM25: Building a Search Engine from Scratch** | Implements TF-IDF (sklearn) and BM25 (rank_bm25) over a corpus of 10K synthetic product listings; evaluates with NDCG@10 and visualizes term-document matrices. |
| 27 | `Build Module 27` | **Word2Vec with Gensim: Query Semantics** | Trains Word2Vec (CBOW & Skip-gram) on synthetic search-query logs; explores nearest-neighbors, analogy tasks, and visualizes embeddings with t-SNE. |
| 28 | `Build Module 28` | **Siamese BERT for Semantic Similarity** | Fine-tunes a small BERT model (via HuggingFace + PyTorch) to score query-document relevance; uses contrastive loss and evaluates on a synthetic labeled relevance dataset. |
| 29 | `Build Module 29` | **Two-Tower Neural Network for Recommendation Retrieval** | Builds a user-tower and item-tower in PyTorch; trains with in-batch negatives on synthetic user-item interaction logs and measures recall@K. |
| 30 | `Build Module 30` | **Matrix Factorization (SVD) for Collaborative Filtering** | Implements ALS and SGD-based matrix factorization on a synthetic movie-rating dataset; evaluates RMSE and visualizes latent factors. |
| 31 | `Build Module 31` | **CTR Prediction with Logistic Regression** | Builds a baseline click-through-rate model on synthetic ad-impression logs; covers feature hashing, calibration (Isotonic/Platt), and online-learning with `SGDClassifier`. |
| 32 | `Build Module 32` | **Wide & Deep Network for Ad Clicks (PyTorch)** | Implements Google's Wide & Deep architecture in PyTorch; compares against logistic regression and a pure deep model on synthetic CTR data. |
| 33 | `Build Module 33` | **Learning to Rank with RankNet (PyTorch)** | Implements pairwise RankNet loss in PyTorch; trains on synthetic query-document pairs and evaluates with NDCG@5 and MAP. |
| 34 | `Build Module 34` | **Multi-Armed Bandit: Epsilon-Greedy for News Headlines** | Simulates a multi-armed bandit environment for headline selection; implements ε-greedy, UCB, and Thompson Sampling; plots cumulative regret curves. |
| 35 | `Build Module 35` | **MapReduce Simulation with Pandas** | Simulates Map and Reduce phases using `groupby` and `apply` on 10M synthetic web-server log rows; demonstrates the shuffle-sort pattern and word-count at scale. |
| 36 | `Build Module 36` | **Approximate Nearest Neighbors with FAISS** | Indexes 1M synthetic item embeddings with FAISS (flat, IVF, HNSW); benchmarks recall vs. speed trade-offs and integrates with the two-tower model from Module 29. |
| 37 | `Build Module 37` | **Trending Topics: LDA & BERTopic on Query Logs** | Applies LDA (Gensim) and BERTopic to synthetic search-query logs; visualizes topic distributions, temporal trends, and builds a topic-dashboard with Plotly. |

---

## Track 4 — Marketing Analytics & Optimization (15 Notebooks)

| # | Command | Title | Description |
|---|---------|-------|-------------|
| 38 | `Build Module 38` | **RFM Segmentation with Pandas** | Computes Recency, Frequency, Monetary metrics on synthetic e-commerce data; segments customers via KMeans and visualizes clusters with a 3-D scatter plot. |
| 39 | `Build Module 39` | **Cohort Analysis: Retention Heatmaps** | Builds weekly and monthly cohort retention tables; visualizes with Seaborn heatmaps and computes average retention curves with confidence bands. |
| 40 | `Build Module 40` | **Customer Lifetime Value: Gamma-Gamma & BG/NBD Models** | Implements the Gamma-Gamma monetary model and BG/NBD frequency model using `lifetimes`; predicts 12-month CLV and validates against holdout actuals. |
| 41 | `Build Module 41` | **Marketing Mix Modeling (MMM) with Regression** | Fits a multi-variate regression (with ad-stock transformations and diminishing-returns curves) to attribute revenue to TV, Social, Search, and Email spend. |
| 42 | `Build Module 42` | **Budget Allocation Optimization with Linear Programming** | Uses `scipy.optimize.linprog` and `minimize` to allocate a fixed marketing budget across channels; maximizes predicted revenue under min/max spend constraints. |
| 43 | `Build Module 43` | **Multi-Touch Attribution with Markov Chains** | Builds a first-order Markov-chain attribution model on synthetic multi-touch journey data; compares with last-click, first-click, and Shapley-value attribution. |
| 44 | `Build Module 44` | **Uplift Modeling: Two-Model Approach (Sklearn)** | Implements the two-model (separate treatment/control models) uplift approach; evaluates with Qini curve and AUUC on a synthetic campaign-response dataset. |
| 45 | `Build Module 45` | **Uplift Modeling: T-Learner with PyTorch** | Replaces the Sklearn base learners with small PyTorch networks; compares uplift predictions and explores calibration of treatment-effect estimates. |
| 46 | `Build Module 46` | **Churn Prevention Campaign: High-Value User Targeting** | Combines a churn-probability model with CLV predictions to prioritize a retention campaign; simulates the campaign outcome and measures incremental revenue. |
| 47 | `Build Module 47` | **Network Effect Analysis: Viral K-Factor** | Calculates the K-factor from synthetic referral-chain data; models viral growth curves and identifies the tipping point where organic growth exceeds paid acquisition. |
| 48 | `Build Module 48` | **Marketing Calendar Visualization with Plotly** | Builds an interactive Plotly heatmap/Gantt showing spend vs. revenue over 52 weeks; overlays campaign events, holidays, and seasonality markers. |
| 49 | `Build Module 49` | **Geo-Experimentation: Difference-in-Differences (DiD)** | Implements a DiD estimator to evaluate a regional marketing campaign; validates parallel-trends assumption and computes the treatment effect with bootstrap CIs. |
| 50 | `Build Module 50` | **Synthetic Data Generation for Privacy Compliance** | Uses `Faker` and copula-based sampling to generate GDPR-safe synthetic user profiles that preserve marginal distributions and correlations of the original data. |
| 51 | `Build Module 51` | **Price Elasticity & Demand Curves** | Fits log-log demand models and uses PyTorch to build a neural demand curve; computes price elasticity of demand and simulates revenue-maximizing prices. |
| 52 | `Build Module 52` | **Capstone: End-to-End Fintech + Marketing Dashboard** | Combines fraud scores, churn predictions, CLV, and campaign attribution into a unified Plotly Dash-style dashboard; demonstrates executive storytelling with data. |

---

## Directory Layout

```
DataPracticum/
├── COURSE_STRUCTURE.md
├── track1_foundations/
│   ├── 01_optimized_pandas.ipynb
│   ├── 02_advanced_groupby.ipynb
│   ├── 03_time_series_resampling.ipynb
│   ├── 04_pytorch_tensors_vs_numpy.ipynb
│   ├── 05_custom_dataloader.ipynb
│   ├── 06_statistical_tests_ab.ipynb
│   ├── 07_imbalanced_data.ipynb
│   ├── 08_hyperparameter_tuning.ipynb
│   ├── 09_feature_selection.ipynb
│   └── 10_sklearn_pipeline.ipynb
├── track2_revolut_fintech/
│   ├── 11_rule_based_fraud.ipynb
│   ├── 12_isolation_forest.ipynb
│   ├── 13_logistic_regression_fraud.ipynb
│   ├── 14_weighted_cross_entropy.ipynb
│   ├── 15_autoencoders_anomaly.ipynb
│   ├── 16_time_series_forecasting.ipynb
│   ├── 17_lstm_dau_prediction.ipynb
│   ├── 18_customer_churn.ipynb
│   ├── 19_graph_neural_networks.ipynb
│   ├── 20_rolling_features.ipynb
│   ├── 21_ab_testing_ui.ipynb
│   ├── 22_credit_risk_scoring.ipynb
│   ├── 23_shap_explainability.ipynb
│   └── 24_data_drift_psi.ipynb
├── track3_yandex_search/
│   ├── 25_text_preprocessing.ipynb
│   ├── 26_tfidf_bm25_search.ipynb
│   ├── 27_word2vec_embeddings.ipynb
│   ├── 28_siamese_bert.ipynb
│   ├── 29_two_tower_retrieval.ipynb
│   ├── 30_matrix_factorization.ipynb
│   ├── 31_ctr_logistic_regression.ipynb
│   ├── 32_wide_and_deep.ipynb
│   ├── 33_learning_to_rank.ipynb
│   ├── 34_multi_armed_bandit.ipynb
│   ├── 35_mapreduce_simulation.ipynb
│   ├── 36_faiss_ann.ipynb
│   └── 37_trending_topics.ipynb
└── track4_marketing/
    ├── 38_rfm_segmentation.ipynb
    ├── 39_cohort_analysis.ipynb
    ├── 40_customer_lifetime_value.ipynb
    ├── 41_marketing_mix_modeling.ipynb
    ├── 42_budget_optimization.ipynb
    ├── 43_markov_attribution.ipynb
    ├── 44_uplift_two_model.ipynb
    ├── 45_uplift_pytorch.ipynb
    ├── 46_churn_prevention_campaign.ipynb
    ├── 47_viral_kfactor.ipynb
    ├── 48_marketing_calendar.ipynb
    ├── 49_geo_did.ipynb
    ├── 50_synthetic_data_privacy.ipynb
    ├── 51_price_elasticity.ipynb
    └── 52_capstone_dashboard.ipynb
```

---

## Notebook Standard Template

Every notebook follows this exact structure:

```
§1  Business Context        — Why this problem matters (with dollar figures / KPIs)
§2  Setup & Imports         — All packages, seeds (np.random.seed(42), torch.manual_seed(42))
§3  Synthetic Data Generation — Realistic data built with numpy + pandas (no external files)
§4  Exploratory Data Analysis — Distributions, correlations, missing-value treatment
§5  Feature Engineering     — Domain-specific features with clear rationale
§6  Modeling                — Classic ML (sklearn) first, then PyTorch for complex tasks
§7  Evaluation              — Business-centric metrics (PR-AUC, NDCG, ROI, etc.)
§8  Visualization           — Matplotlib / Seaborn / Plotly charts for stakeholders
§9  Key Business Takeaway   — One-cell summary with actionable recommendation
```

---

## Prerequisites

```bash
pip install pandas numpy scipy scikit-learn torch statsmodels matplotlib seaborn plotly \
            xgboost lightgbm shap networkx faker lifetimes rank-bm25 gensim \
            transformers optuna faiss-cpu
```

---

## Ready to Build

Type the name of any module (e.g. **`Build Module 1`**) and the complete, self-contained Jupyter Notebook will be generated.

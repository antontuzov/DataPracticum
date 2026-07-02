# Data Science Practicum — Complete Course

> **52 Self-Contained Jupyter Notebooks** covering Data Science Foundations, Fintech, Search & Recommendations, and Marketing Analytics.

## Course Overview

This comprehensive data science course provides hands-on, production-ready notebooks with **synthetic data generation** (no external files needed). Each module follows a standardized template with real-world business context, complete implementations, and actionable takeaways.

### Target Domains
- **Fintech (Revolut)**: Fraud detection, credit risk, churn prediction
- **Big Tech & Search (Yandex)**: Search engines, recommendations, ad targeting
- **Digital Marketing**: Attribution, CLV, uplift modeling, budget optimization

### Tech Stack
Python • Pandas • NumPy • Scikit-learn • PyTorch • Statsmodels • Matplotlib • Seaborn • Plotly • XGBoost • LightGBM • SHAP • NetworkX • Gensim • Transformers • FAISS

---

## Track 1 — Data Science Foundations (10 Notebooks)

| # | Module | Description |
|---|--------|-------------|
| 01 | [Optimized Pandas](track1_foundations/01_optimized_pandas.ipynb) | Vectorization vs. loops vs. apply; `.eval()`, `.query()`, NumPy broadcasting |
| 02 | [Advanced GroupBy](track1_foundations/02_advanced_groupby.ipynb) | Multi-level pivot tables, `agg`, `transform`, rolling/expanding windows |
| 03 | [Time-Series Resampling](track1_foundations/03_time_series_resampling.ipynb) | OHLC candles, Bollinger Bands, RSI indicators with Plotly |
| 04 | [PyTorch Tensors vs NumPy](track1_foundations/04_pytorch_tensors_vs_numpy.ipynb) | GPU transfer, autograd basics, when to choose each library |
| 05 | [Custom DataLoader](track1_foundations/05_custom_dataloader.ipynb) | PyTorch Dataset/DataLoader, batching, multi-worker loading |
| 06 | [Statistical Tests for A/B Testing](track1_foundations/06_statistical_tests_ab.ipynb) | t-tests, Mann-Whitney U, Chi-Squared, bootstrap CIs, power analysis |
| 07 | [Imbalanced Data](track1_foundations/07_imbalanced_data.ipynb) | SMOTE, ADASYN, Tomek Links, class weights, PR-AUC evaluation |
| 08 | [Hyperparameter Tuning](track1_foundations/08_hyperparameter_tuning.ipynb) | GridSearch, Random Search, Bayesian optimization with Optuna |
| 09 | [Feature Selection](track1_foundations/09_feature_selection.ipynb) | Mutual Information, LASSO, SHAP importance, RFE |
| 10 | [Sklearn Pipeline](track1_foundations/10_sklearn_pipeline.ipynb) | Production-grade pipelines, cross-validation, joblib persistence |

---

## Track 2 — Revolut Fintech Specialization (14 Notebooks)

| # | Module | Description |
|---|--------|-------------|
| 11 | [Rule-Based Fraud](track2_revolut_fintech/11_rule_based_fraud.ipynb) | Velocity bursts, geo-impossibility, round-amount flags |
| 12 | [Isolation Forest](track2_revolut_fintech/12_isolation_forest.ipynb) | Anomaly detection, t-SNE visualization, contamination tuning |
| 13 | [Logistic Regression Fraud](track2_revolut_fintech/13_logistic_regression_fraud.ipynb) | Odds ratios, threshold selection, business cost matrix |
| 14 | [Weighted Cross-Entropy](track2_revolut_fintech/14_weighted_cross_entropy.ipynb) | Custom PyTorch loss, imbalanced fraud dataset |
| 15 | [Autoencoders Anomaly](track2_revolut_fintech/15_autoencoders_anomaly.ipynb) | Reconstruction error, unsupervised anomaly detection |
| 16 | [SARIMAX Forecasting](track2_revolut_fintech/16_sarimax_forecasting.ipynb) | Daily transaction volume, Prophet comparison, MAPE evaluation |
| 17 | [LSTM DAU Prediction](track2_revolut_fintech/17_lstm_dau_prediction.ipynb) | Attention mechanism, teacher forcing, gradient clipping |
| 18 | [Churn Prediction](track2_revolut_fintech/18_churn_prediction.ipynb) | XGBoost, SHAP explanations, risk tier segmentation |
| 19 | [Graph Neural Networks](track2_revolut_fintech/19_graph_neural_networks.ipynb) | Money-mule ring detection, NetworkX, circular flow analysis |
| 20 | [Rolling Features](track2_revolut_fintech/20_rolling_features.ipynb) | Real-time feature engineering, incremental updates |
| 21 | [A/B Testing UI](track2_revolut_fintech/21_ab_testing_ui.ipynb) | Frequentist vs Bayesian, power analysis, minimum detectable effect |
| 22 | [Credit Risk Scoring](track2_revolut_fintech/22_credit_risk_scoring.ipynb) | GBM, probability calibration, Platt scaling |
| 23 | [SHAP Explainability](track2_revolut_fintech/23_shap_explainability.ipynb) | Waterfall plots, force plots, compliance explanations |
| 24 | [Data Drift PSI](track2_revolut_fintech/24_data_drift_psi.ipynb) | Population Stability Index, model degradation monitoring |

---

## Track 3 — Yandex Big Tech & Search (13 Notebooks)

| # | Module | Description |
|---|--------|-------------|
| 25 | [Text Preprocessing](track3_yandex_search/25_text_preprocessing.ipynb) | Tokenization, stemming, lemmatization, stopword removal |
| 26 | [TF-IDF & BM25](track3_yandex_search/26_tfidf_bm25_search.ipynb) | Search engine from scratch, NDCG@10 evaluation |
| 27 | [Word2Vec Embeddings](track3_yandex_search/27_word2vec_embeddings.ipynb) | CBOW & Skip-gram, query expansion, t-SNE visualization |
| 28 | [Siamese BERT](track3_yandex_search/28_siamese_bert.ipynb) | Semantic similarity, contrastive loss, HuggingFace |
| 29 | [Two-Tower Retrieval](track3_yandex_search/29_two_tower_retrieval.ipynb) | Recommendation retrieval, in-batch negatives, recall@K |
| 30 | [Matrix Factorization](track3_yandex_search/30_matrix_factorization.ipynb) | SVD, ALS, collaborative filtering, RMSE evaluation |
| 31 | [CTR Prediction](track3_yandex_search/31_ctr_logistic_regression.ipynb) | Feature hashing, calibration, online learning |
| 32 | [Wide & Deep](track3_yandex_search/32_wide_and_deep.ipynb) | Google's architecture, memorization + generalization |
| 33 | [Learning to Rank](track3_yandex_search/33_learning_to_rank.ipynb) | RankNet, pairwise loss, NDCG@K, MAP |
| 34 | [Multi-Armed Bandit](track3_yandex_search/34_multi_armed_bandit.ipynb) | Epsilon-Greedy, UCB, Thompson Sampling, cumulative regret |
| 35 | [MapReduce Simulation](track3_yandex_search/35_mapreduce_simulation.ipynb) | Map/Shuffle/Reduce phases, word count pattern |
| 36 | [FAISS ANN](track3_yandex_search/36_faiss_ann.ipynb) | Approximate nearest neighbors, IVF, HNSW indexes |
| 37 | [Trending Topics](track3_yandex_search/37_trending_topics.ipynb) | LDA, BERTopic, temporal trend analysis |

---

## Track 4 — Marketing Analytics & Optimization (15 Notebooks)

| # | Module | Description |
|---|--------|-------------|
| 38 | [RFM Segmentation](track4_marketing/38_rfm_segmentation.ipynb) | Recency, Frequency, Monetary; KMeans clustering |
| 39 | [Cohort Analysis](track4_marketing/39_cohort_analysis.ipynb) | Retention heatmaps, weekly/monthly cohorts |
| 40 | [Customer Lifetime Value](track4_marketing/40_customer_lifetime_value.ipynb) | Gamma-Gamma & BG/NBD models, 12-month CLV prediction |
| 41 | [Marketing Mix Modeling](track4_marketing/41_marketing_mix_modeling.ipynb) | Adstock transformations, channel attribution, ROI |
| 42 | [Budget Optimization](track4_marketing/42_budget_optimization.ipynb) | Linear programming, diminishing returns, sensitivity analysis |
| 43 | [Markov Attribution](track4_marketing/43_markov_attribution.ipynb) | Multi-touch attribution, removal effect, journey analysis |
| 44 | [Uplift Two-Model](track4_marketing/44_uplift_two_model.ipynb) | Treatment vs control models, Qini curve, AUUC |
| 45 | [Uplift PyTorch](track4_marketing/45_uplift_pytorch.ipynb) | T-Learner with neural networks, heterogeneous treatment effects |
| 46 | [Churn Prevention Campaign](track4_marketing/46_churn_prevention_campaign.ipynb) | Expected loss targeting, CLV-based prioritization |
| 47 | [Viral K-Factor](track4_marketing/47_viral_kfactor.ipynb) | Referral loops, growth modeling, tipping point analysis |
| 48 | [Marketing Calendar](track4_marketing/48_marketing_calendar.ipynb) | Heatmaps, Gantt charts, campaign timeline visualization |
| 49 | [Geo-Experimentation DiD](track4_marketing/49_geo_did.ipynb) | Difference-in-Differences, parallel trends, bootstrap CIs |
| 50 | [Synthetic Data Privacy](track4_marketing/50_synthetic_data_privacy.ipynb) | Faker, GDPR compliance, k-anonymity, correlation preservation |
| 51 | [Price Elasticity](track4_marketing/51_price_elasticity.ipynb) | Demand curves, log-log regression, revenue optimization |
| 52 | [Capstone Dashboard](track4_marketing/52_capstone_dashboard.ipynb) | End-to-end fintech + marketing executive dashboard |

---

## Notebook Template

Every notebook follows this standardized structure:

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

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/DataPracticum.git
   cd DataPracticum
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open any notebook**
   ```bash
   jupyter notebook track1_foundations/01_optimized_pandas.ipynb
   ```

4. **Run all cells** — Each notebook is self-contained with synthetic data generation.

---

## Learning Path

### Beginner Track
Start with **Track 1 (Foundations)** to build core skills:
- Modules 01-03: Pandas mastery
- Modules 04-05: PyTorch basics
- Modules 06-07: Statistical foundations
- Modules 08-10: ML engineering

### Intermediate Track
Apply skills to **Track 2 (Fintech)**:
- Modules 11-15: Fraud detection pipeline
- Modules 16-18: Time series & churn
- Modules 19-24: Advanced techniques

### Advanced Track
Specialize in **Track 3 (Search)** or **Track 4 (Marketing)**:
- Track 3: NLP, recommendations, retrieval systems
- Track 4: Attribution, uplift, optimization

---

## Key Features

- **Self-Contained**: No external data files needed — all generated synthetically
- **Production-Ready**: Real-world patterns, edge cases, and best practices
- **Business-Focused**: Every module includes actionable business takeaways
- **Comprehensive**: From pandas basics to graph neural networks
- **Standardized**: Consistent template across all 52 modules

---

## Use Cases

### For Students
- Learn data science end-to-end
- Build portfolio projects
- Practice with realistic scenarios

### For Instructors
- Ready-to-use course materials
- Consistent structure for grading
- Covers industry-relevant skills

### For Practitioners
- Reference implementations
- Best practices and patterns
- Domain-specific techniques

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by real-world fintech and big tech challenges
- Built with modern Python data science stack
- Designed for practical, hands-on learning

---

**Ready to start?** Pick any module and run it — each notebook is completely self-contained!

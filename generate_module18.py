#!/usr/bin/env python3
"""Generates Module 18: Customer Churn Prediction."""
import json, os

cells = []
def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.split("\n")})
def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "source": s.split("\n"), "outputs": [], "execution_count": None})

md("""# Module 18 — Customer Churn Prediction

> **Track 2 · Revolut Fintech Specialization**  
> **Difficulty**: Intermediate → Advanced  
> **Runtime**: ~5 minutes  
> **Key Libraries**: XGBoost, Scikit-learn, Pandas, NumPy, Plotly, Seaborn  
> **Prerequisite**: Module 10 (Sklearn Pipeline), Module 13 (Logistic Regression)""")

md("""## Table of Contents

1. [Business Context](#1-business-context)
2. [Setup & Imports](#2-setup--imports)
3. [Synthetic Customer Dataset](#3-synthetic-customer-dataset)
4. [Feature Engineering](#4-feature-engineering)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [XGBoost Churn Model](#6-xgboost-churn-model)
7. [Model Evaluation](#7-model-evaluation)
8. [Feature Importance](#8-feature-importance)
9. [Customer Segmentation by Churn Risk](#9-customer-segmentation-by-churn-risk)
10. [Business Impact Analysis](#10-business-impact-analysis)
11. [Visualization Dashboard](#11-visualization-dashboard)
12. [Key Business Takeaway](#12-key-business-takeaway)""")

# ── §1 Business Context ──
md("""---
## §1 · Business Context

### Why Churn Prediction at Revolut?

**Customer churn** is the silent killer of fintech growth:
- **Acquisition cost**: £50–100 per new customer (marketing + KYC).
- **Retention cost**: £5–10 per at-risk customer (push notification, offer).
- **ROI**: Preventing churn is **10× cheaper** than acquiring new customers.

| Churn Signal | Impact |
|--------------|--------|
| **Declining transaction volume** | Customer losing engagement |
| **No login for 30+ days** | Dormant account |
| **Reduced feature usage** | Using only basic features |
| **Negative support interactions** | Frustrated customer |
| **Competitor switching** | Moving funds to Monzo/N26 |

**The business case**:
- Revolut has 30M+ customers.
- 5% monthly churn = 1.5M customers lost/month.
- At £50 acquisition cost = £75M/month in wasted acquisition spend.
- A 10% improvement in retention = £7.5M/month saved.

In this notebook we will:
1. Generate a 100K-customer dataset with engagement and transaction features.
2. Engineer churn-specific features (declining usage, dormancy, feature abandonment).
3. Train an XGBoost model to predict churn probability.
4. Segment customers by churn risk (high/medium/low).
5. Quantify the business impact of a targeted retention campaign.""")

# ── §2 Setup & Imports ──
md("""---
## §2 · Setup & Imports""")

code("""# ── Reproducibility ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import time
import warnings

np.random.seed(42)
warnings.filterwarnings("ignore")

# ── XGBoost ──────────────────────────────────────────────────────
from xgboost import XGBClassifier

# ── Scikit-learn ─────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)

# ── Visualization ────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ── Display settings ─────────────────────────────────────────────
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_theme(style="whitegrid")

print("✓ All imports loaded")""")

# ── §3 Synthetic Data ──
md("""---
## §3 · Synthetic Customer Dataset

We simulate **100,000 customers** with **30 features** spanning:
- **Demographics**: age, tenure, plan type.
- **Transaction behavior**: volume, frequency, recency.
- **Engagement**: login frequency, feature usage, support tickets.
- **Product usage**: number of products, card usage, savings.""")

code("""# ── Configuration ────────────────────────────────────────────────
N_CUSTOMERS = 100_000
CHURN_RATE = 0.08  # 8% monthly churn rate
N_CHURNED = int(N_CUSTOMERS * CHURN_RATE)
N_ACTIVE = N_CUSTOMERS - N_CHURNED

print(f"Generating customer dataset:")
print(f"  Total customers: {N_CUSTOMERS:,}")
print(f"  Churn rate     : {CHURN_RATE*100:.1f}%")

np.random.seed(42)

# ── Feature names ────────────────────────────────────────────────
feature_names = [
    # Demographics
    "age", "tenure_months", "plan_type",  # 0=Basic, 1=Premium, 2=Metal
    # Transaction behavior
    "txn_count_30d", "txn_count_60d", "txn_count_90d",
    "txn_amount_30d", "txn_amount_60d", "txn_amount_90d",
    "txn_frequency", "avg_txn_amount",
    # Engagement
    "login_count_30d", "login_count_60d", "app_sessions_30d",
    "days_since_last_login", "feature_usage_score",
    "support_tickets_30d", "support_satisfaction",
    # Product usage
    "num_products", "has_card", "has_savings", "has_crypto",
    "card_txn_count_30d", "savings_balance",
    # Behavioral signals
    "txn_trend_30d", "login_trend_30d", "feature_trend_30d",
    "competitor_mentions", "negative_reviews",
]

# ── Generate active customers ────────────────────────────────────
X_active = np.random.randn(N_ACTIVE, len(feature_names))
X_active[:, 0] = np.random.normal(35, 10, N_ACTIVE)  # age
X_active[:, 1] = np.random.exponential(12, N_ACTIVE)  # tenure_months
X_active[:, 2] = np.random.choice([0, 1, 2], N_ACTIVE, p=[0.6, 0.3, 0.1])  # plan
X_active[:, 3] = np.random.poisson(15, N_ACTIVE)  # txn_count_30d
X_active[:, 6] = np.abs(np.random.lognormal(4, 1, N_ACTIVE))  # txn_amount_30d
X_active[:, 12] = np.random.poisson(10, N_ACTIVE)  # login_count_30d
X_active[:, 15] = np.random.exponential(3, N_ACTIVE)  # days_since_last_login
X_active[:, 16] = np.random.uniform(50, 100, N_ACTIVE)  # feature_usage_score
X_active[:, 21] = np.random.binomial(1, 0.8, N_ACTIVE)  # has_card

# ── Generate churned customers (different patterns) ──────────────
X_churned = np.random.randn(N_CHURNED, len(feature_names))
X_churned[:, 0] = np.random.normal(30, 12, N_CHURNED)  # younger
X_churned[:, 1] = np.random.exponential(6, N_CHURNED)  # shorter tenure
X_churned[:, 2] = np.random.choice([0, 1, 2], N_CHURNED, p=[0.85, 0.12, 0.03])  # mostly basic
X_churned[:, 3] = np.random.poisson(3, N_CHURNED)  # low txn count
X_churned[:, 6] = np.abs(np.random.lognormal(2.5, 1, N_CHURNED))  # low txn amount
X_churned[:, 12] = np.random.poisson(2, N_CHURNED)  # low login count
X_churned[:, 15] = np.random.exponential(30, N_CHURNED)  # long since last login
X_churned[:, 16] = np.random.uniform(0, 40, N_CHURNED)  # low feature usage
X_churned[:, 17] = np.random.poisson(3, N_CHURNED)  # more support tickets
X_churned[:, 21] = np.random.binomial(1, 0.4, N_CHURNED)  # less likely to have card
X_churned[:, 27] = np.random.normal(-0.5, 0.3, N_CHURNED)  # declining trend

# ── Combine ──────────────────────────────────────────────────────
X = np.vstack([X_active, X_churned])
y = np.concatenate([np.zeros(N_ACTIVE), np.ones(N_CHURNED)])

idx = np.random.permutation(N_CUSTOMERS)
X = X[idx]
y = y[idx]

df = pd.DataFrame(X, columns=feature_names)
df["customer_id"] = np.arange(1, N_CUSTOMERS + 1)
df["churned"] = y

# Round and clean
df["age"] = df["age"].clip(18, 80).round(0).astype(int)
df["tenure_months"] = df["tenure_months"].clip(1, 120).round(0).astype(int)
df["plan_type"] = df["plan_type"].clip(0, 2).round(0).astype(int)
df["days_since_last_login"] = df["days_since_last_login"].clip(0, 90).round(0).astype(int)
df["feature_usage_score"] = df["feature_usage_score"].clip(0, 100).round(1)
df["has_card"] = (df["has_card"] > 0).astype(int)
df["has_savings"] = (df["has_savings"] > 0).astype(int)
df["has_crypto"] = (df["has_crypto"] > 0).astype(int)

print(f"\\n✓ Shape: {df.shape}")
print(f"  Churned: {int(y.sum()):,} ({y.mean()*100:.2f}%)")
print(f"  Active : {int((1-y).sum()):,} ({(1-y).mean()*100:.2f}%)")

df.head()""")

# ── §4 Feature Engineering ──
md("""---
## §4 · Feature Engineering

### Churn-Specific Derived Features""")

code("""# Derived features for churn prediction
df["txn_decline_rate"] = (
    (df["txn_count_30d"] - df["txn_count_60d"] / 2) / (df["txn_count_60d"] / 2 + 1)
).clip(-1, 1)

df["login_decline_rate"] = (
    (df["login_count_30d"] - df["login_count_60d"] / 2) / (df["login_count_60d"] / 2 + 1)
).clip(-1, 1)

df["engagement_score"] = (
    df["login_count_30d"] * 2 +
    df["app_sessions_30d"] +
    df["feature_usage_score"] / 10
) / 10

df["recency_score"] = 100 - df["days_since_last_login"]  # higher = more recent
df["product_depth"] = df["has_card"] + df["has_savings"] + df["has_crypto"]

# Update feature list
all_features = feature_names + [
    "txn_decline_rate", "login_decline_rate",
    "engagement_score", "recency_score", "product_depth",
]

print(f"Total features: {len(all_features)}")
print(f"  Original : {len(feature_names)}")
print(f"  Derived  : 5")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[all_features], df["churned"], test_size=0.3, stratify=df["churned"], random_state=42
)

print(f"\\nTrain: {len(y_train):,} ({y_train.sum():.0f} churned)")
print(f"Test : {len(y_test):,} ({y_test.sum():.0f} churned)")""")

# ── §5 EDA ──
md("""---
## §5 · Exploratory Data Analysis""")

code("""# Key features by churn status
key_features = ["txn_count_30d", "login_count_30d", "days_since_last_login",
                "feature_usage_score", "tenure_months", "engagement_score"]

summary = df.groupby("churned")[key_features].agg(["mean", "median", "std"]).round(2)
print("Feature Statistics by Churn Status:")
print(summary)

print("\\n💡 Churned customers have:")
print("   - Lower transaction and login counts")
print("   - Higher days since last login")
print("   - Lower feature usage and engagement scores")""")

code("""# Correlation with churn
correlations = df[all_features + ["churned"]].corr()["churned"].drop("churned").sort_values()

fig = px.bar(
    x=correlations.values,
    y=correlations.index,
    orientation="h",
    title="Feature Correlation with Churn",
    color=correlations.values,
    color_continuous_scale="RdBu_r",
)
fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
fig.show()

print("💡 Negative correlations = higher values reduce churn")
print("   Positive correlations = higher values increase churn")""")

# ── §6 XGBoost Model ──
md("""---
## §6 · XGBoost Churn Model""")

code("""# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost with class weight balancing
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # handle imbalance
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False,
)

print(f"Training XGBoost Churn Model:")
print(f"  n_estimators    : 200")
print(f"  max_depth       : 6")
print(f"  learning_rate   : 0.1")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

t0 = time.time()
xgb.fit(X_train_scaled, y_train)
train_time = time.time() - t0

print(f"\\n✓ Trained in {train_time:.2f}s")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
print(f"\\n5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")""")

# ── §7 Evaluation ──
md("""---
## §7 · Model Evaluation""")

code("""# Predictions
y_pred = xgb.predict(X_test_scaled)
y_prob = xgb.predict_proba(X_test_scaled)[:, 1]

# Metrics
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 70)
print("XGBOOST CHURN MODEL EVALUATION (Test Set)")
print("=" * 70)
print(f"\\nROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)""")

# ── §8 Feature Importance ──
md("""---
## §8 · Feature Importance""")

code("""# XGBoost feature importance
importance = xgb.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": all_features,
    "Importance": importance,
}).sort_values("Importance", ascending=False)

print("=" * 70)
print("TOP 15 FEATURES FOR CHURN PREDICTION")
print("=" * 70)
print(feature_importance.head(15).to_string(index=False))

print("\\n💡 Top churn predictors:")
print("   - Engagement metrics (login, feature usage)")
print("   - Recency (days since last login)")
print("   - Transaction behavior (declining volume)")""")

code("""# Visualize feature importance
fig = px.bar(
    feature_importance.head(15),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 15 Feature Importances (XGBoost)",
    color="Importance",
    color_continuous_scale="Viridis",
)
fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
fig.show()""")

# ── §9 Customer Segmentation ──
md("""---
## §9 · Customer Segmentation by Churn Risk

### Score All Customers""")

code("""# Predict churn probability for all customers
X_all_scaled = scaler.transform(df[all_features])
churn_prob = xgb.predict_proba(X_all_scaled)[:, 1]
df["churn_probability"] = churn_prob

# Segment into risk tiers
df["risk_tier"] = pd.cut(
    churn_prob,
    bins=[0, 0.2, 0.5, 0.8, 1.0],
    labels=["Low", "Medium", "High", "Critical"],
)

# Summary
risk_summary = df.groupby("risk_tier", observed=True).agg(
    customers=("customer_id", "count"),
    pct=("customer_id", lambda x: len(x) / len(df) * 100),
    avg_prob=("churn_probability", "mean"),
    actual_churn=("churned", "mean"),
).round(4)

print("=" * 80)
print("CUSTOMER SEGMENTATION BY CHURN RISK")
print("=" * 80)
print(risk_summary.to_string())

print("\\n💡 Risk tiers:")
print("   - Low (< 20%): happy customers, minimal intervention")
print("   - Medium (20-50%): at-risk, send engagement emails")
print("   - High (50-80%): likely to churn, offer incentives")
print("   - Critical (> 80%): almost certainly churning, personal outreach")""")

code("""# Visualize risk distribution
fig = px.pie(
    df,
    names="risk_tier",
    title="Customer Distribution by Churn Risk Tier",
    color="risk_tier",
    color_discrete_map={
        "Low": "#00CC96",
        "Medium": "#FFA15A",
        "High": "#EF553B",
        "Critical": "#B7178C",
    },
    hole=0.4,
)
fig.update_layout(height=450)
fig.show()""")

# ── §10 Business Impact ──
md("""---
## §10 · Business Impact Analysis

### Retention Campaign ROI""")

code("""# Business assumptions
ACQUISITION_COST = 75       # £ to acquire a new customer
RETENTION_COST = 10         # £ to retain an at-risk customer (offer, notification)
RETENTION_SUCCESS = 0.30    # 30% of targeted customers stay

# Target high-risk customers
high_risk = df[df["risk_tier"].isin(["High", "Critical"])]
n_targeted = len(high_risk)
n_retained = int(n_targeted * RETENTION_SUCCESS)
n_still_churned = n_targeted - n_retained

# Financial impact
retention_cost = n_targeted * RETENTION_COST
savings = n_retained * ACQUISITION_COST
net_value = savings - retention_cost

print("=" * 80)
print("RETENTION CAMPAIGN BUSINESS IMPACT")
print("=" * 80)

print(f"\\nTargeting:")
print(f"  High/Critical risk customers: {n_targeted:,}")
print(f"  Expected retention rate      : {RETENTION_SUCCESS*100:.0f}%")
print(f"  Customers retained           : {n_retained:,}")

print(f"\\nFinancial Impact:")
print(f"  Retention campaign cost  : £{retention_cost:,.0f}")
print(f"  Savings (avoided churn)  : £{savings:,.0f}")
print(f"  Net value                : £{net_value:,.0f}")
print(f"  ROI                      : {net_value / retention_cost * 100:.0f}%")

print(f"\\n💡 The retention campaign saves £{net_value:,.0f} at {net_value / retention_cost:.1f}× ROI")
print(f"   This is {n_retained * ACQUISITION_COST / (n_targeted * RETENTION_COST):.1f}× more efficient "
      f"than acquiring new customers")""")

code("""# Compare: retention vs. acquisition
print("=" * 80)
print("RETENTION vs. ACQUISITION COST COMPARISON")
print("=" * 80)

print(f"\\nScenario A: Let {n_targeted:,} customers churn, acquire {n_targeted:,} new ones")
cost_acquisition = n_targeted * ACQUISITION_COST
print(f"  Cost: {n_targeted:,} × £{ACQUISITION_COST} = £{cost_acquisition:,.0f}")

print(f"\\nScenario B: Retention campaign (30% success rate)")
cost_retention = retention_cost
print(f"  Cost: {n_targeted:,} × £{RETENTION_COST} = £{cost_retention:,.0f}")

print(f"\\n💡 Retention is {cost_acquisition / cost_retention:.1f}× cheaper than acquisition")
print(f"   Savings: £{cost_acquisition - cost_retention:,.0f}")""")

# ── §11 Dashboard ──
md("""---
## §11 · Visualization Dashboard""")

code("""# ── 11.1 ROC and PR Curves ───────────────────────────────────────
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["ROC Curve", "Precision-Recall Curve"])

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig.add_trace(go.Scatter(
    x=fpr, y=tpr, mode="lines",
    name=f"ROC (AUC = {roc_auc:.4f})",
    line=dict(color="#636EFA", width=2.5),
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    name="Random",
    line=dict(color="gray", width=1, dash="dash"),
), row=1, col=1)

# PR
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
fig.add_trace(go.Scatter(
    x=rec_curve, y=prec_curve, mode="lines",
    name=f"PR (AUC = {pr_auc:.4f})",
    line=dict(color="#EF553B", width=2.5),
), row=1, col=2)

fig.update_layout(height=420, showlegend=True)
fig.show()""")

code("""# ── 11.2 Churn Probability Distribution ──────────────────────────
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=df[df["churned"] == 0]["churn_probability"],
    name="Active Customers",
    nbinsx=100,
    marker_color="#00CC96",
    opacity=0.7,
))

fig.add_trace(go.Histogram(
    x=df[df["churned"] == 1]["churn_probability"],
    name="Churned Customers",
    nbinsx=100,
    marker_color="#EF553B",
    opacity=0.7,
))

fig.update_layout(
    title="Churn Probability Distribution (Active vs. Churned)",
    xaxis_title="Predicted Churn Probability",
    yaxis_title="Count",
    barmode="overlay",
    height=440,
)
fig.show()

print("💡 Good model separation:")
print("   - Active customers cluster at low probability")
print("   - Churned customers cluster at high probability")""")

code("""# ── 11.3 Risk Tier Breakdown ─────────────────────────────────────
risk_counts = df["risk_tier"].value_counts()

fig = go.Figure()

fig.add_trace(go.Bar(
    x=risk_counts.index,
    y=risk_counts.values,
    marker_color=["#00CC96", "#FFA15A", "#EF553B", "#B7178C"],
    text=risk_counts.values,
    textposition="auto",
))

fig.update_layout(
    title="Customer Count by Risk Tier",
    xaxis_title="Risk Tier",
    yaxis_title="Number of Customers",
    height=400,
)
fig.show()

print("💡 The majority of customers are Low risk (healthy)")
print("   High/Critical tiers are the retention campaign targets")""")

# ── §12 Key Business Takeaway ──
md("""---
## §12 · Key Business Takeaway

> ### 🎯 Action Items for a Fintech Data Scientist
>
> | Aspect | Recommendation |
> |--------|---------------|
> | **When to use** | Monthly churn prediction, retention campaign targeting |
> | **Features** | Engagement (login, usage), recency, transaction trends |
> | **Model** | XGBoost with scale_pos_weight for imbalance |
> | **Evaluation** | PR-AUC (not accuracy) for imbalanced churn |
> | **Deployment** | Score all customers monthly, segment into risk tiers |
>
> ### 💡 Revolut-Specific Guidelines
>
> 1. **Score all customers monthly**:
>    ```
>    1st of month → Run churn model → Segment by risk → Trigger campaigns
>    ```
>
> 2. **Tiered retention campaigns**:
>    | Risk Tier | Action | Cost |
>    |-----------|--------|------|
>    | Low | No action | £0 |
>    | Medium | Push notification, feature tips | £2 |
>    | High | Email with personalized offer | £5 |
>    | Critical | Personal call, premium discount | £20 |
>
> 3. **Monitor feature importance monthly**:
>    - Top features change as product evolves.
>    - New features (crypto, stocks) may become churn predictors.
>
> 4. **A/B test retention campaigns**:
>    - Test different offers for High/Critical tiers.
>    - Measure actual churn reduction vs. predicted.
>
> ### 📌 Churn Model Cheat Sheet
>
> ```python
> from xgboost import XGBClassifier
>
> # Handle imbalance
> scale_pos_weight = n_negatives / n_positives
>
> xgb = XGBClassifier(
>     n_estimators=200,
>     max_depth=6,
>     scale_pos_weight=scale_pos_weight,
>     random_state=42,
> )
> xgb.fit(X_train, y_train)
>
> # Score and segment
> churn_prob = xgb.predict_proba(X)[:, 1]
> risk_tier = pd.cut(churn_prob, bins=[0, 0.2, 0.5, 0.8, 1.0],
>                     labels=["Low", "Medium", "High", "Critical"])
> ```
>
> ### 🔑 Key Takeaways
>
> 1. **Retention is 5–10× cheaper than acquisition** — always prioritize churn reduction.
> 2. **Engagement features are the top churn predictors** (login, usage, recency).
> 3. **Segment customers into risk tiers** for targeted retention campaigns.
> 4. **PR-AUC > accuracy** for imbalanced churn prediction.
> 5. **A/B test retention campaigns** to optimize ROI.""")

# ── Build .ipynb ──
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells,
}
for cell in nb["cells"]:
    lines = cell["source"]
    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

out = os.path.join(os.path.dirname(__file__), "track2_revolut_fintech", "18_churn_prediction.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"✓ Notebook written to {out}")
print(f"  Total cells: {len(cells)}")

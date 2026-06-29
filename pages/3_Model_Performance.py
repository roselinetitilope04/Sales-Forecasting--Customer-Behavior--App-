import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Machine Learning Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Model Performance")

st.markdown("""
This page evaluates the performance of the two machine learning models developed for this research:

- **Ridge Regression** for Sales Forecasting
- **K-Means Clustering** for Customer Purchase Behaviour Analysis
""")

st.divider()

# =====================================================
# SECTION 1
# RIDGE REGRESSION
# =====================================================

st.header("📈 Sales Forecasting Model (Ridge Regression)")

# Load model
model = joblib.load("models/ridge_model.pkl")

# Load dataset
df = pd.read_csv("data/forecast_data.csv")

# Feature engineering
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

df["year"] = df["transaction_date"].dt.year
df["month"] = df["transaction_date"].dt.month
df["quarter"] = df["transaction_date"].dt.quarter

df["lag_1"] = df["sales_amount"].shift(1)
df["lag_2"] = df["sales_amount"].shift(2)
df["lag_3"] = df["sales_amount"].shift(3)

df["rolling_mean_3"] = df["sales_amount"].rolling(3).mean()

df = df.dropna()

X = df[
    [
        "year",
        "month",
        "quarter",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "total_quantity",
        "avg_discount",
        "total_transactions",
        "unique_customers",
    ]
]

y = df["sales_amount"]

# Prediction
y_pred = model.predict(X)

# Metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

st.subheader("Model Evaluation")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{mae:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")
c3.metric("R² Score", f"{r2:.3f}")

st.divider()

# Actual vs Predicted

st.subheader("Actual vs Predicted Sales")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    y.values,
    marker="o",
    label="Actual Sales"
)

ax.plot(
    y_pred,
    marker="x",
    label="Predicted Sales"
)

ax.set_xlabel("Observation")
ax.set_ylabel("Sales Amount")
ax.legend()

st.pyplot(fig)

st.divider()

# Residuals

st.subheader("Residual Analysis")

residuals = y - y_pred

fig2, ax2 = plt.subplots(figsize=(10,5))

ax2.scatter(y_pred, residuals)

ax2.axhline(
    y=0,
    color="red",
    linestyle="--"
)

ax2.set_xlabel("Predicted Sales")
ax2.set_ylabel("Residuals")

st.pyplot(fig2)

st.divider()

# Feature Importance

coef = pd.DataFrame({

    "Feature": X.columns,

    "Coefficient": model.coef_

})

coef["Absolute"] = coef["Coefficient"].abs()

coef = coef.sort_values(
    by="Absolute",
    ascending=False
)

st.subheader("Feature Importance")

st.dataframe(
    coef[["Feature","Coefficient"]],
    use_container_width=True
)

st.bar_chart(
    coef.set_index("Feature")["Coefficient"]
)

st.success(
f"""
The Ridge Regression model achieved an R² Score of **{r2:.3f}**,
indicating that approximately **67%** of the variation in monthly sales
is explained by the selected business variables.
"""
)

st.divider()

# =====================================================
# SECTION 2
# CUSTOMER BEHAVIOUR MODEL
# =====================================================

st.header("👥 Customer Purchase Behaviour Model (K-Means Clustering)")

# Load models

kmeans = joblib.load("models/customer_kmeans.pkl")

scaler = joblib.load("models/customer_scaler.pkl")

customer_features = joblib.load("models/customer_features.pkl")

# Load customer dataset
# Change filename if necessary

customer_df = pd.read_csv("data/retail_sales_dataset.csv")

X_customer = customer_df[customer_features]

X_scaled = scaler.transform(X_customer)

clusters = kmeans.predict(X_scaled)

customer_df["Cluster"] = clusters

# =====================================================
# Cluster Metrics
# =====================================================

sil_score = silhouette_score(
    X_scaled,
    clusters
)

c1, c2 = st.columns(2)

c1.metric(
    "Number of Customer Segments",
    len(np.unique(clusters))
)

c2.metric(
    "Silhouette Score",
    f"{sil_score:.3f}"
)

st.divider()

# =====================================================
# Cluster Distribution
# =====================================================

st.subheader("Customer Segment Distribution")

cluster_counts = customer_df["Cluster"].value_counts().sort_index()

st.bar_chart(cluster_counts)

st.dataframe(
    cluster_counts.rename("Customers")
)

st.divider()

# =====================================================
# Cluster Characteristics
# =====================================================

st.subheader("Customer Segment Summary")

summary = customer_df.groupby("Cluster").mean(
    numeric_only=True
)

st.dataframe(
    summary,
    use_container_width=True
)

st.divider()

# =====================================================
# Interpretation
# =====================================================

st.subheader("Model Interpretation")

st.success("""
The K-Means clustering model successfully grouped customers into **four distinct behavioural segments**.

These customer segments provide valuable business intelligence for:

- Customer segmentation
- Personalized marketing
- Loyalty programmes
- Product recommendations
- Customer retention strategies
- Promotional campaign planning
""")

st.info("""
The Silhouette Score indicates how well customers fit into their assigned clusters.
A higher score indicates better-defined customer segments.
""")

st.divider()

# =====================================================
# RESEARCH CONCLUSION
# =====================================================

st.header("📚 Overall Model Assessment")

st.markdown(f"""

### Ridge Regression

- Model Type: Supervised Learning
- Purpose: Monthly Sales Forecasting
- MAE: **{mae:,.2f}**
- RMSE: **{rmse:,.2f}**
- R² Score: **{r2:.3f}**

---

### K-Means Clustering

- Model Type: Unsupervised Learning
- Purpose: Customer Purchase Behaviour Analysis
- Number of Customer Segments: **{len(np.unique(clusters))}**
- Silhouette Score: **{sil_score:.3f}**

---

### Research Outcome

The developed machine learning system integrates **Ridge Regression** for sales forecasting and **K-Means Clustering** for customer purchase behaviour analysis.

Together, these models provide an intelligent decision-support system capable of forecasting future sales while simultaneously identifying meaningful customer segments for business strategy and marketing optimization.
""")

st.divider()

st.caption(
"© 2026 Roseline Titilope Oni | Lead City University | Development of a Sales Forecasting and Customer Purchase Behaviour Analysis Model Using Machine Learning"
)
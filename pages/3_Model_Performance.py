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
# K-MEANS CUSTOMER BEHAVIOUR MODEL PERFORMANCE
# =====================================================

st.divider()

st.header("👥 Customer Purchase Behaviour Model (K-Means Clustering)")

# Load customer dataset
customer_df = pd.read_csv("data/retail_sales_dataset.csv")

customer_df["transaction_date"] = pd.to_datetime(customer_df["transaction_date"])

# -----------------------------------------------------
# Recreate RFM Dataset
# -----------------------------------------------------

snapshot_date = customer_df["transaction_date"].max() + pd.Timedelta(days=1)

rfm = (
    customer_df.groupby("customer_id")
    .agg(
        Recency=("transaction_date",
                 lambda x: (snapshot_date - x.max()).days),
        Frequency=("transaction_id", "count"),
        Monetary=("sales_amount", "sum")
    )
    .reset_index()
)

# -----------------------------------------------------
# Load trained model and scaler
# -----------------------------------------------------

customer_model = joblib.load("models/customer_kmeans.pkl")
customer_scaler = joblib.load("models/customer_scaler.pkl")

# Scale features
X_customer = rfm[["Recency", "Frequency", "Monetary"]]
X_scaled = customer_scaler.transform(X_customer)

# Predict clusters
rfm["Cluster"] = customer_model.predict(X_scaled)

# -----------------------------------------------------
# Cluster Distribution
# -----------------------------------------------------

st.subheader("Cluster Distribution")

cluster_counts = (
    rfm["Cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)

st.dataframe(cluster_counts.rename("Customers"))

# -----------------------------------------------------
# Cluster Profiles
# -----------------------------------------------------

st.subheader("Average Customer Profile")

cluster_profile = (
    rfm.groupby("Cluster")[
        ["Recency", "Frequency", "Monetary"]
    ]
    .mean()
    .round(2)
)

st.dataframe(cluster_profile, use_container_width=True)

# -----------------------------------------------------
# Interpretation
# -----------------------------------------------------

st.subheader("Business Interpretation")

for cluster in sorted(rfm["Cluster"].unique()):

    profile = cluster_profile.loc[cluster]

    st.markdown(f"### Cluster {cluster}")

    st.write(
        f"""
**Average Recency:** {profile['Recency']:.1f} days

**Average Frequency:** {profile['Frequency']:.1f} purchases

**Average Monetary Value:** ₦{profile['Monetary']:,.2f}
"""
    )

# -----------------------------------------------------
# Cluster Scatter Plot
# -----------------------------------------------------

st.subheader("Customer Segments")

import plotly.express as px

fig = px.scatter(
    rfm,
    x="Frequency",
    y="Monetary",
    color=rfm["Cluster"].astype(str),
    size="Monetary",
    hover_data=["Recency"],
    title="Customer Segments"
)

st.plotly_chart(fig, width="stretch")
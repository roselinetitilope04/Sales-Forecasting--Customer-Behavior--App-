import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Ridge Regression Model Performance")

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("models/ridge_model.pkl")

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("data/forecast_data.csv")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

df["year"] = df["transaction_date"].dt.year
df["month"] = df["transaction_date"].dt.month
df["quarter"] = df["transaction_date"].dt.quarter

df["lag_1"] = df["sales_amount"].shift(1)
df["lag_2"] = df["sales_amount"].shift(2)
df["lag_3"] = df["sales_amount"].shift(3)

df["rolling_mean_3"] = df["sales_amount"].rolling(3).mean()

df = df.dropna()

# ==========================================
# FEATURES
# ==========================================
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

# ==========================================
# PREDICTIONS
# ==========================================
y_pred = model.predict(X)

# ==========================================
# METRICS
# ==========================================
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

st.subheader("📈 Model Evaluation")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{mae:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")
c3.metric("R² Score", f"{r2:.3f}")

st.divider()

# ==========================================
# ACTUAL VS PREDICTED
# ==========================================
st.subheader("Actual vs Predicted Sales")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(y.values, marker="o", label="Actual")
ax.plot(y_pred, marker="x", label="Predicted")

ax.set_xlabel("Observation")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

st.divider()

# ==========================================
# RESIDUAL PLOT
# ==========================================
st.subheader("Residual Errors")

residuals = y - y_pred

fig2, ax2 = plt.subplots(figsize=(10,5))

ax2.scatter(y_pred, residuals)

ax2.axhline(y=0, color="red", linestyle="--")

ax2.set_xlabel("Predicted Sales")
ax2.set_ylabel("Residuals")

st.pyplot(fig2)

st.divider()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coef["Absolute"] = coef["Coefficient"].abs()
coef = coef.sort_values("Absolute", ascending=False)

st.subheader("Feature Importance")

st.dataframe(coef[["Feature", "Coefficient"]], use_container_width=True)

st.bar_chart(
    coef.set_index("Feature")["Coefficient"]
)

st.divider()

# ==========================================
# MODEL INTERPRETATION
# ==========================================
st.subheader("Model Interpretation")

st.success(f"✅ R² Score: {r2:.3f}")

st.info(
    """
The Ridge Regression model explains approximately **67%** of the variation in monthly sales.

This indicates that the model captures most of the sales pattern and is suitable for supporting business decision-making, inventory planning, and sales forecasting.
    """
)

st.divider()

st.caption(
    "© 2026 Roseline Titilope Oni | Sales Forecasting using Machine Learning"
)
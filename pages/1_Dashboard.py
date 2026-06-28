import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("data/forecast_data.csv")
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df = df.sort_values("transaction_date")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("Dashboard Controls")

metric = st.sidebar.selectbox(
    "Select Metric",
    ["sales_amount", "total_quantity"]
)

show_data = st.sidebar.checkbox("Show Raw Dataset")

# ==========================================
# TITLE
# ==========================================
st.title("📈 Sales Forecasting and Customer Purchase Behaviour Analysis")

st.markdown("""
### Machine Learning-Based Decision Support System

This application predicts future sales and provides insights into customer purchasing behaviour using a **Ridge Regression** model.
""")

# ==========================================
# PROJECT INFORMATION
# ==========================================
col1, col2, col3 = st.columns(3)

col1.metric("Model", "Ridge Regression")
col2.metric("R² Score", "0.667")
col3.metric("Dataset Size", f"{len(df)} Records")

st.divider()

# ==========================================
# KPI CARDS
# ==========================================
k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Sales",
    f"${df['sales_amount'].sum():,.0f}"
)

k2.metric(
    "Transactions",
    f"{int(df['total_transactions'].sum()):,}"
)

k3.metric(
    "Customers",
    f"{int(df['unique_customers'].sum()):,}"
)

k4.metric(
    "Average Discount",
    f"{df['avg_discount'].mean():.1f}%"
)

st.divider()

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(
    ["📊 Dashboard",
     "📈 Forecast",
     "📄 Research Summary"]
)

# ==========================================
# DASHBOARD
# ==========================================
with tab1:

    st.subheader("Business Dashboard")

    fig1 = px.line(
        df,
        x="transaction_date",
        y=metric,
        title=f"{metric.replace('_',' ').title()} Trend",
        markers=True
    )

    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        fig2 = px.bar(
            df,
            x="transaction_date",
            y="total_quantity",
            title="Monthly Quantity Sold"
        )

        st.plotly_chart(fig2, use_container_width=True)

    with col2:

        fig3 = px.bar(
            df,
            x="transaction_date",
            y="total_transactions",
            title="Monthly Transactions"
        )

        st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:

        fig4 = px.scatter(
            df,
            x="unique_customers",
            y="sales_amount",
            color="sales_amount",
            size="sales_amount",
            title="Customers vs Sales"
        )

        st.plotly_chart(fig4, use_container_width=True)

    with col4:

        fig5 = px.scatter(
            df,
            x="avg_discount",
            y="sales_amount",
            color="avg_discount",
            size="sales_amount",
            title="Discount vs Sales"
        )

        st.plotly_chart(fig5, use_container_width=True)

# ==========================================
# FORECAST
# ==========================================
with tab2:

    st.subheader("Sales Forecast")

    fig = px.line(
        df,
        x="transaction_date",
        y="sales_amount",
        markers=True,
        title="Historical Monthly Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The next version of this application will generate live sales predictions using the trained Ridge Regression model."
    )

# ==========================================
# RESEARCH SUMMARY
# ==========================================
with tab3:

    st.markdown("""
## 📚 Research Summary

### Objective

Develop a machine learning model capable of forecasting future sales and analysing customer purchasing behaviour.

---

### Machine Learning Model

- Ridge Regression

---

### Model Performance

- **MAE:** 32,659.64
- **RMSE:** 39,874.86
- **R² Score:** 0.667

---

### Key Business Insights

- Lipsticks consistently generated the highest revenue.
- Customers aged **18–24** and **55+** contributed significantly to sales.
- Discounts influenced purchasing behaviour.
- The forecasting model supports inventory planning and marketing decisions.

---

### Business Recommendations

- Prioritize VIP customers with loyalty rewards.
- Increase inventory for high-performing products.
- Use sales forecasts for inventory planning.
- Personalize marketing campaigns.
- Monitor discounts and seasonal demand.
""")

# ==========================================
# RAW DATA
# ==========================================
if show_data:

    st.divider()

    st.subheader("Forecast Dataset")

    st.dataframe(df, use_container_width=True)

# ==========================================
# FOOTER
# ==========================================
st.divider()

st.caption(
    "© 2026 Roseline Titilope Oni | MSc Research Project | Sales Forecasting using Machine Learning"
)
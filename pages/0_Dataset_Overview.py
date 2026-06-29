import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Dataset Overview",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Dataset Overview")

st.markdown("""
This page provides an overview of the retail sales dataset used to develop the
**Sales Forecasting and Customer Purchase Behaviour Analysis Model using Machine Learning.**
""")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/retail_sales_dataset.csv")

# Convert date
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# ==========================================
# DATASET SUMMARY
# ==========================================

st.subheader("Dataset Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Records", f"{len(df):,}")
c2.metric("Variables", len(df.columns))
c3.metric("Unique Customers", df["customer_id"].nunique())
c4.metric("Products", df["product_name"].nunique())

st.divider()

# ==========================================
# DATA PREVIEW
# ==========================================

st.subheader("Dataset Preview")

st.dataframe(df.head(20), use_container_width=True)

st.divider()

# ==========================================
# DATA TYPES
# ==========================================

st.subheader("Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(dtype_df, use_container_width=True)

st.divider()

# ==========================================
# MISSING VALUES
# ==========================================

st.subheader("Missing Values")

missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Column": missing.index,
    "Missing Values": missing.values
})

st.dataframe(missing_df, use_container_width=True)

st.divider()

# ==========================================
# SALES BY CATEGORY
# ==========================================

st.subheader("Sales by Product Category")

category_sales = (
    df.groupby("category")["sales_amount"]
      .sum()
      .reset_index()
      .sort_values("sales_amount", ascending=False)
)

fig = px.bar(
    category_sales,
    x="category",
    y="sales_amount",
    color="sales_amount",
    title="Total Sales by Product Category"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# CUSTOMER SEGMENTS
# ==========================================

st.subheader("Customer Segments")

segment = df["customer_segment"].value_counts().reset_index()

segment.columns = ["Segment", "Count"]

fig = px.pie(
    segment,
    names="Segment",
    values="Count",
    title="Distribution of Customer Segments"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# SALES CHANNEL
# ==========================================

st.subheader("Sales Channel Distribution")

channel = df["sales_channel"].value_counts().reset_index()

channel.columns = ["Channel", "Count"]

fig = px.bar(
    channel,
    x="Channel",
    y="Count",
    color="Count"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# PAYMENT METHODS
# ==========================================

st.subheader("Payment Methods")

payment = df["payment_method"].value_counts().reset_index()

payment.columns = ["Payment Method", "Count"]

fig = px.bar(
    payment,
    x="Payment Method",
    y="Count",
    color="Count"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# REGIONAL SALES
# ==========================================

st.subheader("Regional Sales")

region_sales = (
    df.groupby("region")["sales_amount"]
      .sum()
      .reset_index()
)

fig = px.bar(
    region_sales,
    x="region",
    y="sales_amount",
    color="sales_amount"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# AGE GROUP
# ==========================================

st.subheader("Customer Age Groups")

age = df["customer_age_group"].value_counts().reset_index()

age.columns = ["Age Group", "Count"]

fig = px.bar(
    age,
    x="Age Group",
    y="Count",
    color="Count"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# FOOTER
# ==========================================

st.caption(
    "© 2026 Roseline Titilope Oni | Lead City University | Sales Forecasting and Customer Purchase Behaviour Analysis using Machine Learning"
)
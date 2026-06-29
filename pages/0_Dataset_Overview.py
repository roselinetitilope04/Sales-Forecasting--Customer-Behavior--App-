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

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("data/retail_sales_dataset.csv")

df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# ==========================================
# TITLE
# ==========================================
st.title("📂 Dataset Overview")

st.markdown("""
This page presents the dataset used for developing the machine learning models for:

- **Monthly Sales Forecasting (Ridge Regression)**
- **Customer Purchase Behaviour Analysis (K-Means Clustering)**
""")

st.divider()

# ==========================================
# DATASET SUMMARY
# ==========================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Records", len(df))
c2.metric("Variables", len(df.columns))
c3.metric("Missing Values", int(df.isnull().sum().sum()))
c4.metric("Duplicate Records", int(df.duplicated().sum()))

st.divider()

# ==========================================
# DATA DICTIONARY
# ==========================================
st.subheader("📖 Data Dictionary")

dictionary = pd.DataFrame({
    "Variable":[
        "transaction_id",
        "transaction_date",
        "customer_id",
        "gender",
        "age",
        "product_category",
        "quantity",
        "price_per_unit",
        "discount_pct",
        "sales_amount"
    ],

    "Description":[
        "Unique transaction identifier",
        "Date of transaction",
        "Unique customer identifier",
        "Customer gender",
        "Customer age",
        "Purchased product category",
        "Quantity purchased",
        "Unit selling price",
        "Discount percentage",
        "Total sales amount"
    ]
})

st.dataframe(dictionary, width="stretch")

st.divider()

# ==========================================
# DATA PREVIEW
# ==========================================
st.subheader("Dataset Preview")

rows = st.slider(
    "Number of rows",
    5,
    50,
    10
)

st.dataframe(df.head(rows), width="stretch")

st.divider()

# ==========================================
# STATISTICS
# ==========================================
st.subheader("Descriptive Statistics")

st.dataframe(df.describe(), width="stretch")

st.divider()

# ==========================================
# MISSING VALUES
# ==========================================
st.subheader("Missing Values")

missing = df.isnull().sum().reset_index()

missing.columns = ["Variable","Missing"]

fig = px.bar(
    missing,
    x="Variable",
    y="Missing",
    title="Missing Values by Variable"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# PRODUCT CATEGORY
# ==========================================
st.subheader("Product Category Distribution")

category = df["product_category"].value_counts()

fig = px.bar(
    category,
    x=category.index,
    y=category.values,
    labels={"x":"Product Category","y":"Count"},
    title="Products Purchased"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# GENDER
# ==========================================
st.subheader("Customer Gender Distribution")

gender = df["gender"].value_counts()

fig = px.pie(
    values=gender.values,
    names=gender.index,
    title="Gender Distribution"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# AGE DISTRIBUTION
# ==========================================
st.subheader("Customer Age Distribution")

fig = px.histogram(
    df,
    x="age",
    nbins=20,
    title="Age Distribution"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# MONTHLY SALES
# ==========================================
st.subheader("Monthly Sales Trend")

monthly = (
    df.groupby(
        pd.Grouper(
            key="transaction_date",
            freq="ME"
        )
    )["sales_amount"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly,
    x="transaction_date",
    y="sales_amount",
    markers=True,
    title="Monthly Sales"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ==========================================
# SUMMARY
# ==========================================
st.subheader("Dataset Summary")

st.info("""
The retail sales dataset contains transaction-level information describing customer purchasing behaviour.

It includes customer demographics, product information, quantities purchased, discounts applied, and sales values.

The dataset served as the foundation for developing two machine learning models:

• Ridge Regression for monthly sales forecasting.

• K-Means Clustering for customer purchase behaviour analysis.

Feature engineering techniques including lag variables, rolling averages and RFM metrics were applied prior to model development.
""")

st.divider()

st.caption(
    "© 2026 Roseline Titilope Oni | Lead City University"
)
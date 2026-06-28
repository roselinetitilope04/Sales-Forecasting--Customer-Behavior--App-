import streamlit as st

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting and Customer Purchase Behaviour Analysis")

st.write("""
Welcome to the Sales Forecasting and Customer Purchase Behaviour Analysis application.

This application uses a trained Ridge Regression model to forecast monthly sales and provides insights into customer purchasing behaviour using machine learning.
""")
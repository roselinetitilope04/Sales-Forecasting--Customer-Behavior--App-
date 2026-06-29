import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Customer Behaviour Prediction",
    page_icon="👥",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================
kmeans = joblib.load("models/customer_kmeans.pkl")
features = joblib.load("models/customer_features.pkl")
scaler = joblib.load("models/customer_scaler.pkl")

# ==========================================
# TITLE
# ==========================================
st.title("👥 Customer Behaviour Prediction")

st.markdown("""
This page predicts a customer's purchasing behaviour using a trained
**K-Means Clustering** model based on RFM (Recency, Frequency and Monetary) analysis.
""")

st.divider()

# ==========================================
# USER INPUT
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input(
        "Recency (Days Since Last Purchase)",
        min_value=0,
        value=30
    )

with col2:
    frequency = st.number_input(
        "Frequency (Number of Purchases)",
        min_value=1,
        value=10
    )

with col3:
    monetary = st.number_input(
        "Monetary Value ($)",
        min_value=0.0,
        value=500.0
    )

st.divider()

# ==========================================
# PREDICTION
# ==========================================
if st.button("Predict Customer Behaviour", use_container_width=True):

    input_data = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=features
    )

    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Predict cluster
    cluster = kmeans.predict(scaled_input)[0]

    st.success("Prediction completed successfully!")

    st.metric(
        "Predicted Customer Segment",
        f"Cluster {cluster}"
    )

    # ======================================
    # Cluster Interpretation
    # ======================================
    if cluster == 0:
        st.info("""
        **Segment:** Loyal Customers

        These customers purchase frequently and generate high revenue.
        They should be rewarded through loyalty programmes and exclusive offers.
        """)

    elif cluster == 1:
        st.warning("""
        **Segment:** At-Risk Customers

        These customers have not purchased recently.
        Target them with personalised promotions and reminders.
        """)

    elif cluster == 2:
        st.success("""
        **Segment:** High-Value Customers

        These customers contribute significantly to sales.
        Focus on retention and premium customer experiences.
        """)

    else:
        st.error("""
        **Segment:** New or Occasional Customers

        These customers purchase infrequently.
        Encourage repeat purchases using discounts and onboarding campaigns.
        """)

    st.subheader("Input Summary")
    st.dataframe(input_data)

st.divider()

st.caption(
    "© 2026 Roseline Titilope Oni | Customer Behaviour Prediction using K-Means Clustering"
)
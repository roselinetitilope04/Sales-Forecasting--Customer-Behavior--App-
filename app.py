import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sales Forecasting Decision Support System",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================
st.title("📈 Development of a Sales Forecasting and Customer Purchase Behaviour Analysis Model Using Machine Learning")

st.markdown("---")

# ==========================================
# RESEARCH INFORMATION
# ==========================================
col1, col2 = st.columns([2,1])

with col1:

    st.subheader("Research Project")

    st.markdown("""
### Project Title

**Development of a Sales Forecasting and Customer Purchase Behaviour Analysis Model Using Machine Learning**

---

### Researcher

**Roseline Titilope Oni**

### Matric Number

**LCU/PG/0010457**

### Institution

**Lead City University**

### Programme

**Postgraduate Diploma**

### Research Area

Machine Learning • Business Analytics • Sales Forecasting • Customer Behaviour Analysis
""")

with col2:

    st.info("""
### Machine Learning Models

✅ Ridge Regression

• Monthly Sales Forecasting

---

✅ K-Means Clustering

• Customer Behaviour Analysis

---

Interactive Decision Support System
""")

st.divider()

# ==========================================
# OBJECTIVE
# ==========================================
st.header("🎯 Research Objective")

st.write("""
The primary objective of this research is to develop a machine learning-based
decision support system capable of forecasting future sales and analysing
customer purchasing behaviour.

The system assists business managers in making data-driven decisions related to:

- Sales forecasting
- Inventory planning
- Customer segmentation
- Marketing strategies
- Business growth
""")

st.divider()

# ==========================================
# APPLICATION MODULES
# ==========================================
st.header("🗂 Application Modules")

col1, col2 = st.columns(2)

with col1:

    st.success("📊 Dashboard")

    st.write("""
Business intelligence dashboard showing:

- Sales trends
- KPIs
- Monthly transactions
- Customer insights
- Interactive visualisations
""")

    st.success("🔮 Sales Prediction")

    st.write("""
Predict future monthly sales using the trained Ridge Regression model.
""")

    st.success("👥 Customer Behaviour")

    st.write("""
Predict customer purchasing behaviour using the K-Means clustering model.
""")

with col2:

    st.success("📈 Model Performance")

    st.write("""
Evaluate the machine learning models using:

- MAE
- RMSE
- R² Score
- Residual analysis
- Feature importance
""")

    st.success("💼 Business Insights")

    st.write("""
Generate business recommendations from the analysed data.
""")

    st.success("📚 Research Summary")

    st.write("""
Summarises the research methodology, findings, conclusions and recommendations.
""")

st.divider()

# ==========================================
# HOW TO USE
# ==========================================
st.header("🚀 Navigation Guide")

st.info("""
Use the **left sidebar** to navigate through the application.

Recommended order:

1️⃣ Dashboard

2️⃣ Sales Prediction

3️⃣ Customer Behaviour

4️⃣ Model Performance

5️⃣ Business Insights

6️⃣ Research Summary
""")

st.divider()

# ==========================================
# PROJECT HIGHLIGHTS
# ==========================================
st.header("⭐ Project Highlights")

c1, c2, c3 = st.columns(3)

c1.metric("Machine Learning Models", "2")

c2.metric("Decision Support Modules", "6")

c3.metric("Research Status", "Completed")

st.divider()

# ==========================================
# FOOTER
# ==========================================
st.caption(
    "© 2026 Roseline Titilope Oni | Lead City University | "
    "Development of a Sales Forecasting and Customer Purchase Behaviour Analysis Model Using Machine Learning"
)
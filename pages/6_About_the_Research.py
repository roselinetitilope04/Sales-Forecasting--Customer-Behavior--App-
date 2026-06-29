import streamlit as st

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="About the Research", page_icon="📖", layout="wide")

# ==========================
# TITLE
# ==========================
st.title("📖 About the Research")

st.markdown("""
# Development of a Sales Forecasting and Customer Purchase Behaviour Analysis Model Using Machine Learning
""")

st.divider()

# ==========================
# PROJECT SUMMARY
# ==========================
st.header(" Research Overview")

st.write("""
This research focuses on the development of a machine learning-based decision support system for sales forecasting and customer purchase behaviour analysis using historical retail transaction data.

The project combines predictive analytics and customer segmentation to support business decision-making, improve sales planning, and enhance customer understanding.
""")

st.divider()

# ==========================
# OBJECTIVES
# ==========================
st.header(" Research Objectives")

st.markdown("""
- Collect and preprocess historical retail sales data  
- Perform exploratory data analysis (EDA)  
- Engineer predictive features for machine learning models  
- Develop a **Sales Forecasting model using Ridge Regression**  
- Segment customers using **K-Means Clustering**  
- Analyse customer purchase behaviour patterns  
- Identify high-value customer groups  
- Evaluate model performance using MAE, RMSE, and R²  
- Provide business insights and recommendations  
""")

st.divider()

# ==========================
# MODELS USED
# ==========================
st.header("🤖 Machine Learning Models Used")

st.markdown("""
### 1. Ridge Regression (Sales Forecasting Model)
- Used for predicting future monthly sales
- Handles multicollinearity in features using L2 regularization
- Provides stable and interpretable predictions

### 2. K-Means Clustering (Customer Behaviour Model)
- Used to segment customers based on purchasing behaviour
- Groups customers into clusters such as high-value, medium-value, and low-value segments
- Helps businesses understand customer patterns and targeting strategies
""")

st.divider()

# ==========================
# METHODOLOGY
# ==========================
st.header(" Methodology")

st.markdown("""
The project follows the CRISP-DM framework:

1. Business Understanding  
2. Data Understanding  
3. Data Cleaning & Preprocessing  
4. Exploratory Data Analysis (EDA)  
5. Feature Engineering  
6. Model Development (Ridge Regression + K-Means)  
7. Model Evaluation  
8. Interpretation of Results  
9. Deployment using Streamlit  
""")

st.divider()

# ==========================
# EVALUATION
# ==========================
st.header("📊 Model Evaluation Metrics")

st.markdown("""
**Sales Forecasting Model (Ridge Regression):**
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

**Customer Segmentation Model (K-Means):**
- Elbow Method (Inertia)  
- Cluster Interpretation  
""")

st.divider()

# ==========================
# TECHNOLOGIES
# ==========================
st.header("💻 Technologies Used")

st.markdown("""
- Python  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Plotly  
- Streamlit  
- Joblib  
""")

st.divider()

# ==========================
# BUSINESS IMPACT
# ==========================
st.header(" Business Impact")

st.success("""
This system helps organizations to:

• Forecast future sales accurately  
• Improve inventory planning  
• Identify profitable customer segments  
• Enhance marketing targeting strategies  
• Improve customer retention  
• Support data-driven decision-making  
""")

st.divider()

# ==========================
# RESEARCHER INFO
# ==========================
st.header(" Researcher Information")

st.markdown("""
**Name:** Roseline Titilope Oni  
**Matric Number:** LCU/PG/0010457  
**Institution:** Lead City University  
**Department:** Computer Science  
**Level:** Postgraduate (PGD / 600Level)  
""")

st.divider()

# ==========================
# FINAL NOTE
# ==========================
st.info("""
This application is part of a postgraduate research project titled:

**Development of a Sales Forecasting and Customer Purchase Behaviour Analysis Model Using Machine Learning**

It demonstrates how machine learning can be applied to real-world business forecasting and customer segmentation.
""")

st.caption("© 2026 Roseline Titilope Oni | Sales Forecasting & Customer Analytics System")
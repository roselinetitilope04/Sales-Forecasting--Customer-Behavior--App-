import streamlit as st

st.set_page_config(page_title="About the Research", page_icon="📖")

st.title("📖 About the Research")

st.markdown("""
# Development of a Sales Forecasting and Customer Purchase Behaviour Analysis Model Using Machine Learning
""")

st.divider()

st.header("Research Overview")

st.write("""
This research presents the development of a machine learning-based model for forecasting future sales and analysing customer purchase behaviour using historical retail transaction data.

The study integrates data preprocessing, exploratory data analysis, feature engineering, customer segmentation, machine learning, and business intelligence to generate actionable insights that support strategic decision-making.
""")

st.divider()

st.header("Research Objectives")

st.markdown("""
- Collect, clean and preprocess historical sales data.
- Analyse historical sales trends.
- Develop a machine learning model for sales forecasting.
- Analyse customer purchase behaviour.
- Segment customers based on purchasing characteristics.
- Identify high-value customer segments.
- Evaluate model performance using MAE, RMSE and R².
- Generate business recommendations for decision makers.
""")

st.divider()

st.header("Methodology")

st.markdown("""
The project followed the CRISP-DM framework:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Exploratory Data Analysis
5. Feature Engineering
6. Machine Learning Model Development
7. Model Evaluation
8. Business Recommendations
9. Application Deployment
""")

st.divider()

st.header("Machine Learning Model")

st.markdown("""
**Algorithm Used**

- Ridge Regression

**Evaluation Metrics**

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The Ridge Regression model was selected because it provided strong predictive performance while reducing overfitting through regularization.
""")

st.divider()

st.header("Technologies Used")

st.markdown("""
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
""")

st.divider()

st.header("Business Impact")

st.success("""
The developed application enables organizations to:

• Forecast future sales

• Improve inventory planning

• Identify valuable customers

• Support marketing decisions

• Improve customer retention

• Optimize promotional campaigns

• Support data-driven decision making
""")

st.divider()

st.header("Researcher")

st.markdown("""
**Roseline Titilope Oni**

Postgraduate Researcher

Lead City University

Department of Computer Science

Specialization:
- Data Science
- Machine Learning
- Predictive Analytics
""")

st.divider()

st.info("""
This application accompanies the research project titled:

**Development of a Sales Forecasting and Customer Purchase Behaviour Analysis Model Using Machine Learning**

The application demonstrates how machine learning can support sales forecasting and customer behaviour analysis through an interactive decision-support system.
""")

st.divider()

st.caption(
    "© 2026 Roseline Titilope Oni | Sales Forecasting using Machine Learning"
)
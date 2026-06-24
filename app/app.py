import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .header {
        background-color: #1a1a2e;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .header h1 { color: white; font-size: 2rem; }
    .header p { color: rgba(255,255,255,0.6); font-size: 1rem; }
    .result-box {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin-top: 1rem;
    }
    .price { font-size: 3rem; font-weight: bold; color: #1a1a2e; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1>🏠 House Price Predictor</h1>
        <p>Enter house details below to get an estimated sale price</p>
    </div>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_artifacts():
    with open('src/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('src/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('data/train_features_unscaled.csv')
    return df.drop('SalePrice', axis=1)

model, scaler = load_artifacts()
X_template = load_data()

# Inputs
st.subheader("House Details")
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 6000, 1500)
    garage_cars = st.selectbox("Garage Cars", [0, 1, 2, 3, 4])
    total_bsmt = st.number_input("Total Basement (sq ft)", 0, 3000, 800)

with col2:
    year_built = st.number_input("Year Built", 1900, 2024, 2005)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4])
    yr_sold = st.selectbox("Year Sold", [2008, 2009, 2010])
    year_remod = st.number_input("Year Remodelled", 1900, 2024, 2005)

# Predict
if st.button("Predict Price", use_container_width=True):

    # Start with median of unscaled data
    input_df = pd.DataFrame([X_template.median()], columns=X_template.columns)

    # Fill user values
    input_df['OverallQual'] = overall_qual
    input_df['GrLivArea'] = gr_liv_area
    input_df['GarageCars'] = garage_cars
    input_df['TotalBsmtSF'] = total_bsmt
    input_df['YearBuilt'] = year_built
    input_df['FullBath'] = full_bath
    input_df['YrSold'] = yr_sold
    input_df['YearRemodAdd'] = year_remod
    input_df['TotalSF'] = total_bsmt + gr_liv_area
    input_df['HouseAge'] = yr_sold - year_built
    input_df['IsRemodeled'] = int(year_remod != year_built)
    input_df['TotalBath'] = full_bath

    # Scale the input
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=X_template.columns)

    # Predict
    log_price = model.predict(input_scaled_df)[0]
    predicted_price = np.expm1(log_price)

    # Show result
    st.markdown(f"""
        <div class="result-box">
            <p style="color: gray; font-size: 1rem;">Estimated Sale Price</p>
            <p class="price">${predicted_price:,.0f}</p>
            <p style="color: gray; font-size: 0.85rem;">
                Based on XGBoost model &nbsp;·&nbsp; R² = 88.9%
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                <div><b>88.9%</b><br><small>R² Score</small></div>
                <div><b>0.14</b><br><small>RMSE</small></div>
                <div><b>XGBoost</b><br><small>Model</small></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
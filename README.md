# 🏠 House Price Prediction

A machine learning web app that predicts house sale prices using the Ames Housing dataset.

## 🔍 Project Overview
- Performed EDA on 80+ features
- Cleaned missing values and applied log transformation on target
- Engineered new features: TotalSF, HouseAge, IsRemodeled, TotalBath
- Trained and compared Linear Regression, Ridge, and XGBoost
- Best model: XGBoost with R² = 88.9%
- Explained predictions using SHAP values
- Deployed as interactive Streamlit web app

## 📊 Model Performance
| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | 87.46% | 0.153 |
| Ridge | 87.53% | 0.152 |
| XGBoost | 88.93% | 0.143 |

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- SHAP
- Streamlit

## 📁 Project Structure
house-price-prediction/

├── data/

├── notebooks/

│   ├── 01_eda.ipynb

│   ├── 02_data_cleaning.ipynb

│   ├── 03_feature_engineering.ipynb

│   ├── 04_model_building.ipynb

│   └── 05_shap.ipynb

├── src/

│   ├── model.pkl

│   └── scaler.pkl

├── app/

│   └── app.py

└── requirements.txt

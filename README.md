# Airbnb Price Prediction — MLCOE Task 3

## Project Overview

This project was completed as part of MLCOE Task 3 on Unsupervised Learning and Ensemble Techniques.

The project uses a real-world Airbnb dataset containing more than 150,000 rows and 30+ features. The main goal is to analyze the data, apply PCA, study clustering concepts, train ensemble regression models, and deploy the final model using Streamlit.

## Dataset

The dataset was sourced from Kaggle.

Target variable:
- `price`

Problem type:
- Regression

## Project Workflow

1. Exploratory Data Analysis
   - Dataset structure and data types
   - Missing values and duplicate analysis
   - Distribution and outlier analysis
   - Correlation analysis
   - Feature engineering

2. PCA
   - Standardization of numerical features
   - Principal Component Analysis
   - Cumulative explained variance analysis
   - Selection of components based on explained variance

3. Unsupervised Learning — Conceptual Study
   - K-Means Clustering
   - DBSCAN

4. Ensemble Learning
   - Random Forest Regressor
   - AdaBoost Regressor
   - XGBoost Regressor
   - Hyperparameter experimentation
   - Model comparison
   - Feature importance analysis

5. Model Export
   - Saved Random Forest model using Joblib
   - Saved preprocessing information required during prediction

6. Streamlit Deployment
   - Interactive user inputs
   - Live Airbnb price prediction

## Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 66.32 | 108.39 | 0.6944 |
| XGBoost | 69.47 | 111.91 | 0.6742 |
| AdaBoost | 129.52 | 162.07 | 0.3167 |

The Random Forest model achieved the strongest results in the current comparison based on the evaluation metrics.

## Live Demo

[Airbnb Price Prediction App](https://task03-unsupervised-learning-k5ftuu5rz8mhgwy6gdapty.streamlit.app/)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## Project Files

- `Task-3.ipynb` — complete analysis and model development
- `app.py` — Streamlit application
- `random_forest_model.pkl` — trained model
- `feature_columns.pkl` — model feature structure
- `feature_medians.pkl` — preprocessing values
- `cities.pkl` — city values for the application
- `property_types.pkl` — property type values
- `requirements.txt` — required Python packages

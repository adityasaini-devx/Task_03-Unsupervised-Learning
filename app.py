import streamlit as st
import pandas as pd
import joblib

model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
medians = joblib.load("feature_medians.pkl")
cities = joblib.load("cities.pkl")
property_types = joblib.load("property_types.pkl")

st.title("Airbnb Price Prediction")
st.write("Enter the property details to predict its price.")

latitude = st.number_input("Latitude", value=40.7)
longitude = st.number_input("Longitude", value=-73.9)

accommodates = st.number_input("Accommodates", min_value=1, value=2)
bedrooms = st.number_input("Bedrooms", min_value=0, value=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, value=1.0)
beds = st.number_input("Beds", min_value=0, value=1)

city = st.selectbox("City", cities)
property_type = st.selectbox("Property Type", property_types)

input_data = pd.DataFrame([medians])
input_data["latitude"] = latitude
input_data["longitude"] = longitude
input_data["accommodates"] = accommodates
input_data["bedrooms"] = bedrooms
input_data["bathrooms"] = bathrooms
input_data["beds"] = beds

city_col = "city_" + city

if city_col in input_data.columns:
    input_data[city_col] = 1

property_col = "property_type_" + property_type

if property_col in input_data.columns:
    input_data[property_col] = 1

input_data = input_data.reindex(columns=feature_columns, fill_value=0)

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Price: {prediction:.2f}")
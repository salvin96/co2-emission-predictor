import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Title
st.title("🌍 CO₂ Emission Prediction App")
st.write("Powered by Random Forest | Interactive & Free")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

# Load scaler
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# Sidebar inputs
st.sidebar.header("Input Features")

coal_co2 = st.sidebar.slider("Coal CO₂ (Mt)", 0.0, 50000.0, 10000.0)
oil_co2 = st.sidebar.slider("Oil CO₂ (Mt)", 0.0, 50000.0, 10000.0)
gdp = st.sidebar.slider("GDP (in trillions)", 0.0, 30.0, 15.0)
population = st.sidebar.slider("Population (in billions)", 0.0, 10.0, 5.0)
year = st.sidebar.slider("Year", 1950, 2025, 2020)

# Prepare raw input
input_df = pd.DataFrame({
    "coal_co2": [coal_co2],
    "oil_co2": [oil_co2],
    "gdp": [gdp],
    "population": [population],
    "year": [year]
})

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict CO₂ Emissions"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"🌱 Predicted CO₂ Emission: {prediction:.2f} Megatons")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Team 3 AML")

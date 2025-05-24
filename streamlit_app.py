
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Title
st.title("🌍 CO₂ Emission Prediction App")
st.write("Powered by Random Forest | Interactive & Free")

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()

# Define input features (customize based on your dataset)
st.sidebar.header("Input Features")

# Example inputs - to be replaced by actual features in the dataset
feature_1 = st.sidebar.slider("GDP (in trillions)", 0.0, 30.0, 15.0)
feature_2 = st.sidebar.slider("Energy Consumption (TWh)", 0.0, 50000.0, 25000.0)
feature_3 = st.sidebar.slider("Population (in billions)", 0.0, 10.0, 5.0)

# Prepare input DataFrame
input_df = pd.DataFrame({
    "gdp": [feature_1],
    "energy_consumption": [feature_2],
    "population": [feature_3]
})

# Prediction
if st.button("Predict CO₂ Emissions"):
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Predicted CO₂ Emission: {prediction:.2f} Megatons")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Team 3")

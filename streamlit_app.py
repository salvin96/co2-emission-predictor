import streamlit as st
from streamlit_lottie import st_lottie
import json
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="CO2 Predictor", layout="centered")

# Load model and scaler
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl"), joblib.load("scaler.pkl")

model, scaler = load_model()

# Background animation using custom HTML (Lottie background-like effect)
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://assets6.lottiefiles.com/packages/lf20_hbr24n7k.json");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        section.main > div {
            background-color: rgba(255, 255, 255, 0.92);
            padding: 2rem;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Page title
st.title("🌍 CO₂ Emission Prediction App")
st.write("Powered by Random Forest | Interactive & Free")

# Input fields with colored sliders
coal_co2 = st.slider("🪨 Coal CO₂ (Mt)", 0.0, 4000.0, 500.0, key="coal", label_visibility="visible")
st.markdown('<style>div[data-baseweb="slider"] > div:first-child { background-color: orange; }</style>', unsafe_allow_html=True)
st.caption("Emissions from coal-based energy and industry.")

oil_co2 = st.slider("🛢️ Oil CO₂ (Mt)", 0.0, 4000.0, 1000.0, key="oil", label_visibility="visible")
st.markdown('<style>div[data-baseweb="slider"]:nth-of-type(2) > div:first-child { background-color: steelblue; }</style>', unsafe_allow_html=True)
st.caption("Emissions from petroleum-based sources.")

gdp = st.slider("💰 GDP (Trillions)", 0.0, 30.0, 15.0, key="gdp", label_visibility="visible")
st.markdown('<style>div[data-baseweb="slider"]:nth-of-type(3) > div:first-child { background-color: green; }</style>', unsafe_allow_html=True)
st.caption("Gross Domestic Product (economic output).")

population = st.slider("👥 Population (Billions)", 0.0, 1.5, 0.7, key="population", label_visibility="visible")
st.markdown('<style>div[data-baseweb="slider"]:nth-of-type(4) > div:first-child { background-color: purple; }</style>', unsafe_allow_html=True)
st.caption("Population of the country or region.")

year = st.slider("📅 Year", 1950, 2025, 2020, key="year", label_visibility="visible")
st.caption("Year of prediction context.")

# Prediction
input_data = np.array([[coal_co2, oil_co2, gdp, population, year]])
scaled_data = scaler.transform(input_data)
if st.button("Predict CO₂ Emissions"):
    result = model.predict(scaled_data)[0]
    if result < 3000:
        st.success(f"🟢 Predicted CO₂ Emission: {result:.2f} Megatons")
    elif result < 6000:
        st.warning(f"🟡 Predicted CO₂ Emission: {result:.2f} Megatons")
    else:
        st.error(f"🔴 Predicted CO₂ Emission: {result:.2f} Megatons")

# Footer
st.markdown("""
    <hr style='border: 1px solid #ddd;'>
    <center>Created with ❤️ by Team 3</center>
""", unsafe_allow_html=True)

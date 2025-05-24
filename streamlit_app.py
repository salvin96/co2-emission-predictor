import streamlit as st
import pandas as pd
import joblib
import numpy as np
from streamlit_lottie import st_lottie
import requests

# Load Lottie animation from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation for welcome
lottie_login = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_yr6zz3wv.json")

# Login check
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.username:
    st.markdown("## 👤 Login Required")
    st_lottie(lottie_login, speed=1, height=250, key="welcome_lottie")

    username = st.text_input("Enter your name to begin:", key="username_input")
    if st.button("Start"):
        if username.strip():
            st.session_state.username = username.strip().title()
            st.rerun()
        else:
            st.warning("Please enter a valid name.")
    st.stop()

# Load model and scaler
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# Title with user's name
st.title(f"🌍 CO₂ Emission Prediction App | Welcome, {st.session_state.username}!")

# Sidebar inputs
st.sidebar.header("Input Features")

coal_co2 = st.sidebar.slider("Coal CO₂ (Mt)", 0.0, 4000.0, 1000.0)
oil_co2 = st.sidebar.slider("Oil CO₂ (Mt)", 0.0, 4000.0, 1000.0)
gdp = st.sidebar.slider("GDP (in trillions)", 0.0, 30.0, 15.0)
population = st.sidebar.slider("Population (in billions)", 0.0, 1.5, 0.7)
year = st.sidebar.slider("Year", 1950, 2025, 2020)

# Prepare input DataFrame
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

    # CO₂ Classification
    if prediction < 3000:
        st.success(f"🟢 CO₂ Emission: {prediction:.2f} Megatons — **Green Zone** (Safe)")
    elif 3000 <= prediction <= 6000:
        st.warning(f"🟡 CO₂ Emission: {prediction:.2f} Megatons — **Yellow Zone** (Moderate)")
    else:
        st.error(f"🔴 CO₂ Emission: {prediction:.2f} Megatons — **Red Zone** (High!)")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Team 3")

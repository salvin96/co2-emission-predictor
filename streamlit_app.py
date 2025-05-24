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

# Background image for main content only
st.markdown("""
    <style>
    /* Sidebar stays clean */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    /* Main background image behind content */
    .main::before {
        content: "";
        background: linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)), url("https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/oso.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        z-index: -1;
        filter: brightness(0.95);
    }

    /* Make main content readable */
    section.main > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# Lottie login animation
lottie_login = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_yr6zz3wv.json")

# Session check for login
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

# App title
st.title(f"🌍 CO₂ Emission Prediction App | Welcome, {st.session_state.username}!")

# Sidebar Input Features
st.sidebar.header("🧮 Input Features")

# 🪨 Coal CO₂
st.sidebar.markdown("#### <span style='color:#D2691E'>🪨 Coal CO₂ (Mt)</span>", unsafe_allow_html=True)
coal_co2 = st.sidebar.slider("Coal CO₂", 0.0, 8000.0, 1000.0, label_visibility="collapsed")
st.sidebar.caption("Emissions from coal-based energy and industry.")

# 🛢️ Oil CO₂
st.sidebar.markdown("#### <span style='color:#4682B4'>🛢️ Oil CO₂ (Mt)</span>", unsafe_allow_html=True)
oil_co2 = st.sidebar.slider("Oil CO₂", 0.0, 8000.0, 1000.0, label_visibility="collapsed")
st.sidebar.caption("Emissions from petroleum-based sources.")

# 💰 GDP
st.sidebar.markdown("#### <span style='color:#228B22'>💰 GDP (Trillions)</span>", unsafe_allow_html=True)
gdp = st.sidebar.slider("GDP", 0.0, 30.0, 15.0, label_visibility="collapsed")
st.sidebar.caption("Gross Domestic Product (economic output).")

# 👥 Population
st.sidebar.markdown("#### <span style='color:#8A2BE2'>👥 Population (Billions)</span>", unsafe_allow_html=True)
population = st.sidebar.slider("Population", 0.0, 1.5, 0.7, label_visibility="collapsed")
st.sidebar.caption("Population of the country or region.")

# 📅 Year
st.sidebar.markdown("#### <span style='color:#696969'>📅 Year</span>", unsafe_allow_html=True)
year = st.sidebar.slider("Year", 1950, 2025, 2020, label_visibility="collapsed")
st.sidebar.caption("Year of prediction context.")

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

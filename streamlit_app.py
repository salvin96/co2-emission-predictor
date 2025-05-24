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

# App title
st.title(f"🌍 CO₂ Emission Prediction App | Welcome, {st.session_state.username}!")

# Sidebar input section
st.sidebar.header("🧮 Input Features")

# Sliders with color-coded headers
st.sidebar.markdown("#### <span style='color:#D2691E'>🪨 Coal CO₂ (Mt)</span>", unsafe_allow_html=True)
coal_co2 = st.sidebar.slider("coal", 0.0, 4000.0, 1000.0, label_visibility="collapsed")
st.sidebar.caption("Emissions from coal-based energy and industry.")

st.sidebar.markdown("#### <span style='color:#4682B4'>🛢️ Oil CO₂ (Mt)</span>", unsafe_allow_html=True)
oil_co2 = st.sidebar.slider("oil", 0.0, 4000.0, 1000.0, label_visibility="collapsed")
st.sidebar.caption("Emissions from petroleum-based sources.")

st.sidebar.markdown("#### <span style='color:#228B22'>💰 GDP (Trillions)</span>", unsafe_allow_html=True)
gdp = st.sidebar.slider("gdp", 0.0, 30.0, 15.0, label_visibility="collapsed")
st.sidebar.caption("Gross Domestic Product (economic output).")

st.sidebar.markdown("#### <span style='color:#8A2BE2'>👥 Population (Billions)</span>", unsafe_allow_html=True)
population = st.sidebar.slider("population", 0.0, 1.5, 0.7, label_visibility="collapsed")
st.sidebar.caption("Population of the country or region.")

st.sidebar.markdown("#### <span style='color:#696969'>📅 Year</span>", unsafe_allow_html=True)
year = st.sidebar.slider("year", 1950, 2025, 2020, label_visibility="collapsed")
st.sidebar.caption("Year of prediction context.")

# Style each slider with its color using nth-of-type trick
st.markdown("""
    <style>
        /* Coal CO₂ (1st slider) */
        section[data-testid="stSidebar"] div[data-baseweb="slider"]:nth-of-type(1) > div:first-child {
            background-color: #D2691E !important;
        }
        /* Oil CO₂ (2nd slider) */
        section[data-testid="stSidebar"] div[data-baseweb="slider"]:nth-of-type(2) > div:first-child {
            background-color: #4682B4 !important;
        }
        /* GDP (3rd slider) */
        section[data-testid="stSidebar"] div[data-baseweb="slider"]:nth-of-type(3) > div:first-child {
            background-color: #228B22 !important;
        }
        /* Population (4th slider) */
        section[data-testid="stSidebar"] div[data-baseweb="slider"]:nth-of-type(4) > div:first-child {
            background-color: #8A2BE2 !important;
        }
        /* Year (5th slider) */
        section[data-testid="stSidebar"] div[data-baseweb="slider"]:nth-of-type(5) > div:first-child {
            background-color: #696969 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Prepare input for model
input_df = pd.DataFrame({
    "coal_co2": [coal_co2],
    "oil_co2": [oil_co2],
    "gdp": [gdp],
    "population": [population],
    "year": [year]
})

scaled_input = scaler.transform(input_df)

# Prediction button
if st.button("Predict CO₂ Emissions"):
    prediction = model.predict(scaled_input)[0]

    if prediction < 3000:
        st.success(f"🟢 CO₂ Emission: {prediction:.2f} Megatons — **Green Zone** (Safe)")
    elif 3000 <= prediction <= 6000:
        st.warning(f"🟡 CO₂ Emission: {prediction:.2f} Megatons — **Yellow Zone** (Moderate)")
    else:
        st.error(f"🔴 CO₂ Emission: {prediction:.2f} Megatons — **Red Zone** (High!)")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Team 3")

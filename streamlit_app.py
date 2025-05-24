import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Title
st.title("🌍 CO₂ Emission Prediction App")
st.write("Powered by Random Forest | Interactive & Free")

# Load model and scaler
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

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
    st.success(f"🌱 Predicted CO₂ Emission: {prediction:.2f} Megatons")

    # ========================
    # Visualization Section
    # ========================
    st.markdown("## 📊 Emission Trends & Prediction Visuals")

    # Load historical CO₂ data
    try:
        df_hist = pd.read_excel("owid-co2-data-FINAL-cleaned.xlsx")
        df_hist = df_hist[['year', 'co2']].dropna()
        df_hist = df_hist.groupby('year').sum().reset_index()

        st.line_chart(df_hist.rename(columns={'co2': 'Historical CO₂ Emissions'}).set_index('year'))

        last_year = int(input_df['year'][0])
        new_row = pd.DataFrame({'year': [last_year], 'co2': [prediction]})
        df_combined = pd.concat([df_hist[df_hist['year'] < last_year], new_row], ignore_index=True)
        df_combined = df_combined.sort_values('year')
        st.line_chart(df_combined.rename(columns={'co2': 'Historical + Predicted CO₂'}).set_index('year'))

    except Exception as e:
        st.warning("📉 Could not load historical CO₂ data: " + str(e))

    # Bar chart of inputs
    st.markdown("### 🔍 Feature Inputs Overview")
    st.bar_chart(input_df.T.rename(columns={0: "User Input"}))

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Team 3")

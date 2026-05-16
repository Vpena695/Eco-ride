
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# --- Streamlit App Interface (DEBE ser lo primero) ---
st.set_page_config(
    page_title="Sistema de Alerta Temprana de Churn - Eco-Ride",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Sistema de Alerta Temprana de Churn - Eco-Ride")
st.markdown("Ingrese los datos del cliente para predecir el riesgo de cancelación.")

# --- Load the preprocessor and model ---
# Use st.cache_resource for objects that should be loaded once
@st.cache_resource
def load_resources():
    try:
        # Load the preprocessing pipeline
        pipeline_preproc = joblib.load('pipeline_preproc.pkl')
        # Load the trained churn model
        modelo_churn = joblib.load('modelo_churn.pkl')
        return pipeline_preproc, modelo_churn
    except FileNotFoundError:
        st.error("Error: Make sure 'modelo_churn.pkl' and 'pipeline_preproc.pkl' are in the same directory as this app.py file.")
        st.stop() # Stop the app if essential files are missing

pipeline_preproc, modelo_churn = load_resources()


# Input controls for customer data
st.header("Datos del Cliente")

col1, col2 = st.columns(2)

with col1:
    edad = st.slider("Edad", min_value=18, max_value=80, value=30, help="Edad del cliente.")
    plan_options = ['Básico', 'Premium', 'Elite']
    plan = st.selectbox("Plan", options=plan_options, help="Tipo de plan del cliente.")
    uso_mensual_km = st.slider("Uso Mensual Km", min_value=0.0, max_value=200.0, value=50.0, step=0.1, help="Kilómetros recorridos mensualmente.")

with col2:
    soporte_tickets = st.slider("Soporte Tickets", min_value=0, max_value=10, value=1, help="Número de tickets de soporte generados.")
    gasto_promedio = st.slider("Gasto Promedio", min_value=10.0, max_value=500.0, value=100.0, step=0.1, help="Gasto promedio del cliente.")
    region_options = ['Norte', 'Sur', 'Centro']
    region = st.selectbox("Región", options=region_options, help="Región geográfica del cliente.")

# 'Dias_Antiguedad' was an engineered feature, not directly provided by the user.
# We'll use a default value based on the mean from the training data (approx. 940 days).
# If specific input for this feature is desired, a slider could be added.
dias_antiguedad_default = 940 # Mean 'Dias_Antiguedad' from X_train

# --- Prediction Logic ---
if st.button("Analizar Riesgo"):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'Edad': [edad],
        'Plan': [plan.lower()], # Homogenize 'Plan' to lowercase as done in preprocessing
        'Uso_Mensual_Km': [uso_mensual_km],
        'Soporte_Tickets': [soporte_tickets],
        'Gasto_Promedio': [gasto_promedio],
        'Region': [region],
        'Dias_Antiguedad': [dias_antiguedad_default] # Add the derived feature
    })

    # Apply preprocessing pipeline (only transform)
    # The pipeline expects original column names.
    processed_input = pipeline_preproc.transform(input_data)

    # Make prediction and get probabilities
    prediction = modelo_churn.predict(processed_input)[0]
    prediction_proba = modelo_churn.predict_proba(processed_input)[0] # [proba_no_churn, proba_churn]

    st.subheader("Resultado del Análisis")

    if prediction == 1:
        st.error(f"🚨 **Alto Riesgo de Cancelación**")
        st.write(f"Probabilidad de Cancelación: **{prediction_proba[1]*100:.2f}%**")
    else:
        st.success(f"✅ **Cliente Estable**")
        st.write(f"Probabilidad de Cancelación: **{prediction_proba[1]*100:.2f}%**")

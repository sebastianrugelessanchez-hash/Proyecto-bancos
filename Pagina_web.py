import streamlit as st
import joblib
import numpy as np

# Cargar el modelo entrenado
modelo = joblib.load("xgboost_credit_risk_20251004_163147.pkl")

# Título
st.title("Credit Risk Prediction")

st.write("Esta aplicación predice si eres un riesgo **Good** o **Bad** usando el modelo entrenado en XGBoost.")

# Inputs del usuario
credit_amount = st.number_input("Enter Credit Amount:", min_value=0.0, step=100.0)
housing_rent = st.number_input("Enter Housing Rent:", min_value=0.0, step=50.0)
age = st.number_input("Enter Age:", min_value=18, step=1)

# Botón para predecir
if st.button("Predict Risk"):
    # Crear el array de entrada con los valores ingresados
    input_data = np.array([[credit_amount, housing_rent, age]])

    # Predicción
    prediction = modelo.predict(input_data)
    prediction_proba = modelo.predict_proba(input_data)  # Probabilidades Good/Bad

    # Mostrar resultado
    if prediction[0] == 1:  # asumiendo 1 = Good, 0 = Bad
        st.success(f"Predicted Risk: Good ✅ (Probabilidad: {prediction_proba[0][1]:.2f})")
    else:
        st.error(f"Predicted Risk: Bad ⚠️ (Probabilidad: {prediction_proba[0][0]:.2f})")

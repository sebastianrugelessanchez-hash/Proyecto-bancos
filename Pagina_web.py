import streamlit as st

# Título de la app
st.title("Credit Risk Prediction")

st.write("Esta aplicación predice si eres un riesgo **Good** o **Bad** basándose en tu monto de crédito, renta de vivienda y edad.")

# Inputs del usuario
credit_amount = st.number_input("Enter Credit Amount:", min_value=0.0, step=100.0)
housing_rent = st.number_input("Enter Housing Rent:", min_value=0.0, step=50.0)
age = st.number_input("Enter Age:", min_value=18, step=1)

# Lógica simple para predecir Good o Bad (esto lo puedes reemplazar con un modelo real de ML)
def predict_risk(credit_amount, housing_rent, age):
    score = (age / 100) + (housing_rent / 1000) - (credit_amount / 10000)
    if score > 0:
        return "Good"
    else:
        return "Bad"

# Botón de predicción
if st.button("Predict Risk"):
    prediction = predict_risk(credit_amount, housing_rent, age)
    if prediction == "Good":
        st.success(f"Predicted Risk: {prediction}")
    else:
        st.error(f"Predicted Risk: {prediction}")
import streamlit as st
import joblib
import numpy as np

# =========================
# Cargar modelo y objetos
# =========================
modelo = joblib.load("xgboost_credit_risk_20251004_163147.pkl")
encoder = joblib.load("onehot_encoder_20251004_165417.pkl")
class_weights = joblib.load("balance_info_class_weights.pkl")  # solo para mostrar info

# =========================
# Interfaz de usuario
# =========================
st.title("Credit Risk Prediction")

st.write("Esta aplicación predice si eres un riesgo **Good** o **Bad** usando el modelo entrenado en XGBoost.")
st.write("El modelo utiliza las variables: **Duration, Housing_rent, Credit_amount y Age**.")

# Inputs del usuario
duration = st.number_input("Enter Duration (in months):", min_value=1, step=1)
housing_rent = st.selectbox("Select Housing Rent:", ["own", "rent", "free"])  # categorías que usaste en el entrenamiento
credit_amount = st.number_input("Enter Credit Amount:", min_value=0.0, step=100.0)
age = st.number_input("Enter Age:", min_value=18, step=1)

# =========================
# Predicción
# =========================
if st.button("Predict Risk"):
    # 1. Crear array con los valores
    input_data = np.array([[duration, housing_rent, credit_amount, age]])

    # 2. Aplicar el OneHotEncoder (transforma housing_rent a columnas binarias)
    input_data_encoded = encoder.transform(input_data)

    # 3. Predecir con el modelo
    prediction = modelo.predict(input_data_encoded)
    prediction_proba = modelo.predict_proba(input_data_encoded)

    # 4. Mostrar resultado
    if prediction[0] == 1:  # suponiendo 1 = Good
        st.success(f"Predicted Risk: Good ✅ (Probabilidad: {prediction_proba[0][1]:.2f})")
    else:
        st.error(f"Predicted Risk: Bad ⚠️ (Probabilidad: {prediction_proba[0][0]:.2f})")

    # 5. Mostrar info adicional de los class weights
    st.write("### Class Weights usados en el entrenamiento:")
    st.json(class_weights)



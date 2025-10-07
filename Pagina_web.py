import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import List

st.title("Prediction Application")

# ============== Utilidades ==============
def to_numeric_series(x):
    if isinstance(x, (int, float, np.number)):
        return x
    if pd.isna(x):
        return np.nan
    x = str(x).replace(",", "").replace(" ", "")
    try:
        return float(x)
    except Exception:
        return np.nan

def ensure_columns_for_artifact(X: pd.DataFrame, artifact, defaults: dict = None) -> pd.DataFrame:
    """
    Asegura que X tenga EXACTAMENTE las columnas que el artefacto espera (feature_names_in_).
    Crea las que falten con valores por defecto y reordena.
    """
    defaults = defaults or {}
    if hasattr(artifact, "feature_names_in_"):
        expected = list(artifact.feature_names_in_)
        X2 = X.copy()
        # agrega faltantes
        for c in expected:
            if c not in X2.columns:
                X2[c] = defaults.get(c, 0)
        # elimina extras
        X2 = X2[expected]
        return X2
    return X

def apply_any_encoder(encoder, X: pd.DataFrame) -> pd.DataFrame:
    """
    Maneja OneHotEncoder, ColumnTransformer o Pipeline con OneHot adentro, o LabelEncoder (raro en X, pero por si acaso).
    Devuelve DataFrame con columnas one-hot (o la salida que corresponda) y nombres.
    """
    # Si es LabelEncoder entrenado solo en 'Felder' (vector 1-D)
    from sklearn.preprocessing import LabelEncoder
    if isinstance(encoder, LabelEncoder):
        if "Felder" not in X.columns:
            raise ValueError("LabelEncoder requiere columna 'Felder'.")
        y = X["Felder"].astype(str).values.ravel()
        arr = encoder.transform(y)
        return pd.DataFrame({"Felder_le": arr})

    # ColumnTransformer / Pipeline / OneHotEncoder
    # Antes: nos aseguramos que X tenga las columnas esperadas
    X_ready = ensure_columns_for_artifact(
        X, encoder,
        defaults={c: "__missing__" for c in X.columns if X[c].dtype == "object"}
    )

    Z = encoder.transform(X_ready)

    # Obtén nombres de salida
    feature_names = None
    try:
        # OneHotEncoder y ColumnTransformer modernos
        feature_names = encoder.get_feature_names_out()
    except Exception:
        # OneHotEncoder antiguo
        try:
            cats = getattr(encoder, "categories_", None)
            if cats is not None:
                # Suponemos que solo codificaste 'Felder'
                cats = cats[0]
                feature_names = [f"Felder_{c}" for c in cats]
        except Exception:
            pass

    if hasattr(Z, "toarray"):
        Z = Z.toarray()

    if feature_names is None:
        feature_names = [f"enc_{i}" for i in range(Z.shape[1])]

    return pd.DataFrame(Z, columns=feature_names)

def align_to_model_features(X: pd.DataFrame, model) -> pd.DataFrame:
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "estimators_"):
        try:
            final_est = model.estimators_[-1]
            if hasattr(final_est, "feature_names_in_"):
                expected = list(final_est.feature_names_in_)
        except Exception:
            pass
    if expected is None:
        return X
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    X = X[expected]
    return X

# ============== Carga de artefactos ==============
encoder_file_path = 'onehot_encoder.joblib'
scaler_file_path = 'minmax_scaler.joblib'
model_file_path = 'logistic_regression_best_model.joblib'

try:
    onehot_encoder = joblib.load(encoder_file_path)
    st.write("One-hot encoder loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: onehot_encoder.joblib not found at {encoder_file_path}.")
    st.stop()
except Exception as e:
    st.error(f"Error loading one-hot encoder: {e}")
    st.stop()

try:
    scaler = joblib.load(scaler_file_path)
    st.write("Min-Max scaler loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: minmax_scaler.joblib not found at {scaler_file_path}.")
    st.stop()
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

try:
    best_model = joblib.load(model_file_path)
    st.write("Logistic Regression Model loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: logistic_regression_best_model.joblib not found at {model_file_path}.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ============== UI ==============
felder_input = st.selectbox(
    "Select Felder:",
    ['activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal']
)
examen_admision_input = st.number_input(
    "Enter Examen de admisión Universidad:",
    min_value=0.0, max_value=10.0, step=0.01
)

# ============== Construcción del DataFrame base (mismo NOMBRE que en entrenamiento) ==============
# OJO: estos nombres deben coincidir con los que tu encoder/scaler aprendieron
X_base = pd.DataFrame({
    'Felder': [str(felder_input)],
    'Examen_admisión_Universidad': [to_numeric_series(examen_admision_input)]
})

# ============== ENCODING ==============
try:
    encoded_df = apply_any_encoder(onehot_encoder, X_base[['Felder']])
    st.write("Encoding OK. Shape:", encoded_df.shape)
except Exception as e:
    st.error("Fallo al aplicar el encoder (posible desalineación de columnas o tipo de encoder).")
    st.exception(e)
    st.stop()

# ============== SCALING ==============
scaled_df = pd.DataFrame()
try:
    # Asegura columnas esperadas por el scaler
    X_num = X_base.select_dtypes(include=['number'])
    X_num = ensure_columns_for_artifact(X_num, scaler, defaults={})
    Z = scaler.transform(X_num)
    scaled_df = pd.DataFrame(Z, columns=getattr(scaler, "feature_names_in_", X_num.columns))
    # Si tu modelo espera sufijo _scaled, descomenta:
    # scaled_df.columns = [f"{c}_scaled" for c in scaled_df.columns]
    st.write("Scaling OK. Shape:", scaled_df.shape)
except Exception as e:
    st.error("Fallo al aplicar el scaler (posible desalineación de columnas).")
    st.exception(e)
    st.stop()

# ============== COMBINA Y ALINEA PARA EL MODELO ==============
X_processed_input = pd.concat([scaled_df, encoded_df], axis=1)
X_processed_input = align_to_model_features(X_processed_input, best_model)

st.write("X_processed_input shape:", X_processed_input.shape)
st.write("X_processed_input columns:", list(X_processed_input.columns))

# ============== PREDICCIÓN ==============
if st.button("Predict"):
    try:
        y_pred = best_model.predict(X_processed_input)
        st.subheader("Prediction:")
        st.write(str(y_pred[0]))
    except Exception as e:
        st.error("Error during prediction.")
        st.exception(e)
        # Ayuda de diagnóstico
        if hasattr(best_model, 'feature_names_in_'):
            st.write("Model expected features:", best_model.feature_names_in_.tolist())














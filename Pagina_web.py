import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Prediction Application")

# ========= RUTAS DE ARTEFACTOS (las reales) =========
ENCODER_PATH = "onehot_encoder_20251004_165417.pkl"          # OneHot de features
SCALER_PATH  = "minmax_scaler.joblib"                        # Escalador de numéricas
MODEL_PATH   = "logistic_regression_best_model.joblib"       # Modelo final
Y_LABEL_PATH = "label_encoders_20251004_165417.pkl"          # (OPCIONAL) LabelEncoder del target Risk

# ========= UTILIDADES =========
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", " ", regex=True)
          .str.replace("/", "_", regex=False)
          .str.replace("-", "_", regex=False)
    )
    return df

def soft_to_numeric(s: pd.Series) -> pd.Series:
    # Convierte strings con separadores de miles/comas/espacios a numérico
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(" ", "", regex=False)
         .pipe(pd.to_numeric, errors="ignore")
    )

def ensure_expected_columns(X: pd.DataFrame, artifact, fill_map: dict | None = None) -> pd.DataFrame:
    """Asegura que X tenga exactamente feature_names_in_ si el artefacto lo provee."""
    fill_map = fill_map or {}
    if hasattr(artifact, "feature_names_in_"):
        expected = list(artifact.feature_names_in_)
        X2 = X.copy()
        for c in expected:
            if c not in X2.columns:
                X2[c] = fill_map.get(c, 0)
        X2 = X2[expected]
        return X2
    return X

def align_to_model_features(X: pd.DataFrame, model) -> pd.DataFrame:
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "estimators_"):
        try:
            last_est = model.estimators_[-1]
            if hasattr(last_est, "feature_names_in_"):
                expected = list(last_est.feature_names_in_)
        except Exception:
            pass
    if expected is None:
        return X
    for c in expected:
        if c not in X.columns:
            X[c] = 0
    return X[expected]

# ========= CARGA DE ARTEFACTOS =========
try:
    ohe = joblib.load(ENCODER_PATH)
    st.success("OneHotEncoder loaded.")
except Exception as e:
    st.error(f"No pude cargar el OneHotEncoder: {e}")
    st.stop()

try:
    scaler = joblib.load(SCALER_PATH)
    st.success("MinMaxScaler loaded.")
except Exception as e:
    st.error(f"No pude cargar el scaler: {e}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded.")
except Exception as e:
    st.error(f"No pude cargar el modelo: {e}")
    st.stop()

# (OPCIONAL) LabelEncoder del target para mostrar Good/Bad
y_label = None
try:
    y_label = joblib.load(Y_LABEL_PATH)
    st.info("Target LabelEncoder loaded (optional).")
except Exception:
    y_label = None

# ========= UI: SUBIR CSV =========
st.write("Sube un CSV con los **predictors** del cliente (sin la columna 'Risk').")
file = st.file_uploader("CSV de clientes nuevos", type=["csv"])

if file is not None:
    try:
        raw = pd.read_csv(file)
        st.write("Vista previa:")
        st.dataframe(raw.head())
    except Exception as e:
        st.error(f"Error leyendo el CSV: {e}")
        st.stop()

    # Limpieza rápida
    df = normalize_columns(raw.copy())
    # Normaliza tipos: intenta pasar a numérico todas las columnas que se vean numéricas
    for c in df.columns:
        df[c] = soft_to_numeric(df[c])

    # Separamos columnas esperadas por OHE y por scaler
    # Usamos feature_names_in_ de cada artefacto (si existen)
    ohe_input_cols = list(getattr(ohe, "feature_names_in_", []))
    scaler_input_cols = list(getattr(scaler, "feature_names_in_", []))

    # Si el OHE se entrenó con varias categóricas, esperamos verlas aquí
    X_cat = df[ohe_input_cols] if ohe_input_cols else df.select_dtypes(exclude=["number"])
    # Si el scaler se entrenó con numéricas específicas, usamos esas; si no, tomamos las numéricas del df
    X_num = df[scaler_input_cols] if scaler_input_cols else df.select_dtypes(include=["number"])

    # Asegura columnas esperadas por cada artefacto
    # Para categóricas faltantes, rellenamos con string especial (OneHot con handle_unknown='ignore' lo ignora)
    X_cat = ensure_expected_columns(X_cat, ohe, fill_map={c: "__missing__" for c in (ohe_input_cols or [])})
    X_num = ensure_expected_columns(X_num, scaler, fill_map={})

    # Transformaciones
    try:
        Z_cat = ohe.transform(X_cat)
        if hasattr(Z_cat, "toarray"):
            Z_cat = Z_cat.toarray()
        try:
            cat_cols_out = ohe.get_feature_names_out(ohe_input_cols if ohe_input_cols else X_cat.columns)
        except Exception:
            cat_cols_out = [f"cat_{i}" for i in range(Z_cat.shape[1])]
        df_cat = pd.DataFrame(Z_cat, columns=cat_cols_out, index=df.index)
    except Exception as e:
        st.error("Fallo aplicando el OneHotEncoder. Verifica que el encoder fue entrenado con estas columnas.")
        st.error(str(e))
        st.stop()

    try:
        Z_num = scaler.transform(X_num)
        num_cols_out = getattr(scaler, "feature_names_in_", X_num.columns)
        df_num = pd.DataFrame(Z_num, columns=num_cols_out, index=df.index)
    except Exception as e:
        st.error("Fallo aplicando el scaler. Verifica columnas numéricas esperadas.")
        st.error(str(e))
        st.stop()

    # Combina y alinea con el modelo
    X_proc = pd.concat([df_num, df_cat], axis=1)
    X_proc = align_to_model_features(X_proc, model)

    # Predicción
    try:
        y_pred = model.predict(X_proc)
        # Mapea a Good/Bad si tenemos el LabelEncoder del target
        if y_label is not None:
            try:
                y_pretty = y_label.inverse_transform(y_pred)
            except Exception:
                y_pretty = y_pred
        else:
            y_pretty = y_pred

        out = df.copy()
        out["Prediction"] = y_pretty
        st.subheader("Resultados")
        st.dataframe(out.head(50))
        # Descarga
        out_csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar predicciones (CSV)", data=out_csv, file_name="predicciones_risk.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.write("Columnas procesadas:", list(X_proc.columns))
        if hasattr(model, "feature_names_in_"):
            st.write("Columnas esperadas por el modelo:", list(model.feature_names_in_))















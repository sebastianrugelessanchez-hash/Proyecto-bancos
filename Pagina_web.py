import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("Credit Risk – Inference")

# ===== Sidebar: rutas de artefactos =====
st.sidebar.header("Artefactos")
OHE_PATH        = st.sidebar.text_input("OneHotEncoder", "onehot_encoder_20251004_165417.pkl")
KNN_CAT_PATH    = st.sidebar.text_input("KNN categóricas", "knn_imputer_categorical_20251004_165417.pkl")
KNN_NUM_PATH    = st.sidebar.text_input("KNN numéricas", "knn_imputer_numerical_20251004_165417.pkl")
LE_FEATS_PATH   = st.sidebar.text_input("LabelEncoders (features)", "label_encoders_20251004_165417.pkl")
MODEL_PATH      = st.sidebar.text_input("Modelo XGBoost", "xgboost_credit_risk_20251004_163147.pkl")

# Mapeo de salida (si no tienes LabelEncoder del target)
st.sidebar.subheader("Salida")
map_output = st.sidebar.checkbox("Mapear 0/1 → Good/Bad", value=True)
label_for_0 = st.sidebar.text_input("Etiqueta para 0", "Good")
label_for_1 = st.sidebar.text_input("Etiqueta para 1", "Bad")

with st.expander("📁 Diagnóstico"):
    st.write("cwd:", os.getcwd())
    try: st.write(os.listdir())
    except Exception as e: st.write(e)

# ===== Carga de artefactos (no detenemos la UI) =====
artifacts_ok = True
def _load(path, name):
    global artifacts_ok
    try:
        obj = joblib.load(path)
        st.success(f"{name} cargado.")
        return obj
    except Exception as e:
        st.error(f"No pude cargar {name} en '{path}': {e}")
        artifacts_ok = False
        return None

ohe         = _load(OHE_PATH, "OneHotEncoder")
knn_cat     = _load(KNN_CAT_PATH, "KNN Imputer (categóricas)")
knn_num     = _load(KNN_NUM_PATH, "KNN Imputer (numéricas)")
le_features = _load(LE_FEATS_PATH, "LabelEncoders de features")
model       = _load(MODEL_PATH, "Modelo")

# ===== Utilidades =====
def normalize_columns(df):
    df.columns = (df.columns.str.strip()
                  .str.replace(r"\s+", " ", regex=True)
                  .str.replace("/", "_", regex=False)
                  .str.replace("-", "_", regex=False))
    return df

def soft_to_numeric(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
            .pipe(pd.to_numeric, errors="ignore"))

def align_to_model_features(X: pd.DataFrame, model) -> pd.DataFrame:
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    if expected is None:
        return X
    for c in expected:
        if c not in X.columns: X[c] = 0
    return X[expected]

def encode_for_knn_cat(values: pd.Series, le) -> pd.Series:
    classes = set(le.classes_)
    def _map(v):
        if pd.isna(v) or str(v) not in classes:
            return np.nan
        return le.transform([str(v)])[0]
    return values.astype(object).map(_map)

def inverse_clip(series_num: pd.Series, le) -> pd.Series:
    arr = series_num.round().astype(int)
    arr = arr.clip(0, len(le.classes_) - 1)
    return pd.Series(le.inverse_transform(arr), index=series_num.index)

# ===== UI: carga de CSV =====
st.write("Sube un CSV de **nuevos clientes** (sin la columna 'Risk').")
file = st.file_uploader("CSV", type=["csv"])

if file is not None:
    try:
        raw = pd.read_csv(file)
        st.write("Vista previa:")
        st.dataframe(raw.head())
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")

    if artifacts_ok:
        df = normalize_columns(raw.copy())
        for c in df.columns:
            df[c] = soft_to_numeric(df[c])

        # --- Imputación categóricas ---
        if knn_cat is not None and isinstance(le_features, dict):
            cat_cols = list(getattr(knn_cat, "feature_names_in_", le_features.keys()))
            for c in cat_cols:
                if c not in df.columns: df[c] = np.nan
            enc_df = pd.DataFrame(index=df.index)
            for c in cat_cols:
                le = le_features.get(c)
                enc_df[c] = encode_for_knn_cat(df[c].astype(str), le) if le else np.nan
            enc_imputed = pd.DataFrame(knn_cat.transform(enc_df), columns=cat_cols, index=df.index)
            for c in cat_cols:
                le = le_features.get(c)
                if le: df[c] = inverse_clip(enc_imputed[c], le)
            st.info("Imputación KNN (categóricas) ok.")

        # --- Imputación numéricas ---
        if knn_num is not None:
            num_cols = list(getattr(knn_num, "feature_names_in_", df.select_dtypes(include="number").columns))
            for c in num_cols:
                if c not in df.columns: df[c] = np.nan
            num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
            num_imputed = pd.DataFrame(knn_num.transform(num_df), columns=num_cols, index=df.index)
            df[num_cols] = num_imputed
            st.info("Imputación KNN (numéricas) ok.")

        # --- OneHot ---
        ohe_in_cols = list(getattr(ohe, "feature_names_in_", []))
        if not ohe_in_cols:
            ohe_in_cols = [c for c in df.columns if df[c].dtype == "object"]
        for c in ohe_in_cols:
            if c not in df.columns: df[c] = "__missing__"
            df[c] = df[c].fillna("__missing__").astype(str)

        Z_cat = ohe.transform(df[ohe_in_cols])
        if hasattr(Z_cat, "toarray"): Z_cat = Z_cat.toarray()
        try:
            cat_out_cols = ohe.get_feature_names_out(ohe_in_cols)
        except Exception:
            cat_out_cols = [f"cat_{i}" for i in range(Z_cat.shape[1])]
        df_cat = pd.DataFrame(Z_cat, columns=cat_out_cols, index=df.index)

        # Numéricas finales = todas las numéricas ya imputadas que no van al OHE
        X_num = df.drop(columns=ohe_in_cols, errors="ignore").select_dtypes(include=["number"])
        X = pd.concat([X_num, df_cat], axis=1)
        X = align_to_model_features(X, model)

        # --- Predicción ---
        if st.button("Predict"):
            try:
                y = model.predict(X)
                if map_output:
                    mapping = {0: label_for_0, 1: label_for_1}
                    y_pretty = [mapping.get(int(v), str(v)) for v in y]
                else:
                    y_pretty = y
                out = raw.copy()
                out["Prediction"] = y_pretty
                st.subheader("Resultados")
                st.dataframe(out.head(50))
                st.download_button("Descargar (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_risk.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error durante la predicción: {e}")
                st.write("Cols procesadas:", list(X.columns))
                if hasattr(model, "feature_names_in_"):
                    st.write("Cols esperadas por el modelo:", list(model.feature_names_in_))
    else:
        st.warning("Faltan artefactos. Corrige las rutas en la barra lateral para habilitar la predicción.")

















import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Credit Risk Scoring", page_icon="💳", layout="centered")

# =============================
# Sidebar: artefactos y modo
# =============================
st.sidebar.title("⚙️ Configuración del Modelo")
mode = st.sidebar.radio("Cómo cargar artefactos", ["Usar rutas locales", "Subir archivos"], index=0)

MODEL_OBJ = None
ENCODER_OBJ = None
CLASSWEIGHTS_OBJ = None

def safe_unpickle(file_like):
    """Intenta cargar con pickle y, si falla, con joblib."""
    try:
        return pickle.load(file_like), None
    except Exception:
        try:
            import joblib
            # reposicionar el puntero por si el primer intento leyó bytes
            file_like.seek(0)
            return joblib.load(file_like), None
        except Exception as e2:
            return None, f"Error cargando archivo: {e2}"

if mode == "Usar rutas locales":
    # 🔁 RUTAS RELATIVAS a la raíz del repo (donde está Pagina_web.py)
    model_path = st.sidebar.text_input("Ruta del modelo (.pkl)", "./xgboost_credit_risk_20251004_163147.pkl")
    enc_path   = st.sidebar.text_input("Ruta OneHotEncoder (opcional)", "./onehot_encoder_20251004_165417.pkl")
    wts_path   = st.sidebar.text_input("Ruta class weights (opcional)", "./balance_info_class_weights.pkl")

    def load_if_exists(path):
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return safe_unpickle(f)
        return (None, f"No encontrado: {path}")

    MODEL_OBJ, m_err = load_if_exists(model_path)
    ENCODER_OBJ, e_err = load_if_exists(enc_path) if enc_path else (None, None)
    CLASSWEIGHTS_OBJ, w_err = load_if_exists(wts_path) if wts_path else (None, None)
else:
    st.sidebar.caption("Sube tus artefactos entrenados:")
    up_model = st.sidebar.file_uploader("Modelo (.pkl)", type=["pkl"], key="mdl")
    up_enc   = st.sidebar.file_uploader("OneHotEncoder (.pkl, opcional)", type=["pkl"], key="enc")
    up_wts   = st.sidebar.file_uploader("Class Weights (.pkl, opcional)", type=["pkl"], key="wts")

    if up_model is not None:
        MODEL_OBJ, m_err = safe_unpickle(up_model)
    else:
        MODEL_OBJ, m_err = (None, "Falta el modelo")
    ENCODER_OBJ, e_err = safe_unpickle(up_enc) if up_enc is not None else (None, None)
    CLASSWEIGHTS_OBJ, w_err = safe_unpickle(up_wts) if up_wts is not None else (None, None)

for err in [m_err, e_err, w_err]:
    if err:
        st.sidebar.warning(err)

# =============================
# Helpers
# =============================
def fallback_predict(df: pd.DataFrame):
    """Regla simple con bonus por categorías (ilustrativo; NO usar como sustituto del modelo real)."""
    credit_amount = df.get("credit_amount", pd.Series([0]*len(df))).astype(float)
    housing_rent  = df.get("housing_rent", pd.Series([0]*len(df))).astype(float)
    age           = df.get("age", pd.Series([30]*len(df))).astype(float)
    housing = df.get("housing", pd.Series(["other"]*len(df))).astype(str).str.lower()
    job     = df.get("job", pd.Series(["other"]*len(df))).astype(str).str.lower()
    purpose = df.get("purpose", pd.Series(["other"]*len(df))).astype(str).str.lower()

    cat_bonus = (
        housing.isin(["own"]).astype(int)*0.10
        + job.isin(["skilled","management"]).astype(int)*0.10
        + purpose.isin(["education","furniture/equipment"]).astype(int)*0.05
    )

    score = (age/100.0) + (housing_rent/1000.0) - (credit_amount/10000.0) + cat_bonus
    proba = (score - score.min()) / (score.max() - score.min() + 1e-6)
    yhat  = np.where(score > 0, "Good", "Bad")
    return pd.DataFrame({"score_fallback": score, "proba_good_fallback": proba, "prediction": yhat})

# === MAPEOS SEGUROS (NUEVO) ============================================
# Normaliza nombres de columnas para coincidir con el entrenamiento.
# Cubre mayúsculas, espacios finales, snake_case y el typo "Purpuso".
RENAME_MAP_SAFE = {
    # variantes → nombre entrenado
    "housing": "Housing",
    "Housing": "Housing",

    "purpose": "Purpose",
    "Purpuso": "Purpose",          # typo
    "Purpose": "Purpose",

    "saving accounts": "Saving accounts",
    "Saving accounts": "Saving accounts",
    "Saving accounts ": "Saving accounts",   # con espacio colgante
    "saving_accounts": "Saving accounts",

    "checking account": "Checking account",
    "Checking account": "Checking account",
    "checking_account": "Checking account",

    # numéricas por si vinieran con otro estilo
    "age": "Age",
    "Age": "Age",
    "job": "Job",
    "Job": "Job",
    "credit amount": "Credit amount",
    "Credit amount": "Credit amount",
    "credit_amount": "Credit amount",
    "duration": "Duration",
    "Duration": "Duration",
}

CAT_TRAIN_COLS = ["Checking account", "Saving accounts", "Housing", "Purpose"]
NUM_TRAIN_COLS = ["Age", "Job", "Credit amount", "Duration"]

def normalize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve df con nombres corregidos al esquema de entrenamiento."""
    # strip en nombres
    cols = [c.strip() for c in df.columns]
    df = df.copy()
    df.columns = cols
    # aplica mapeos directos
    df = df.rename(columns=RENAME_MAP_SAFE)
    # garantiza presencia de columnas esperadas
    for c in CAT_TRAIN_COLS:
        if c not in df.columns:
            df[c] = None
    for c in NUM_TRAIN_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df
# =======================================================================

def apply_encoder(enc, X: pd.DataFrame):
    """Aplica OneHotEncoder si existe; concatena numéricas + codificadas."""
    if enc is None:
        return X, []
    try:
        # (NUEVO) normaliza antes de detectar categóricas
        X = normalize_feature_names(X)

        cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
        if not cat_cols:
            return X, []
        # asegura dtype category para estabilidad
        for c in cat_cols:
            X[c] = X[c].astype("category")

        Xt = enc.transform(X[cat_cols])
        try:
            colnames = enc.get_feature_names_out(cat_cols)
        except Exception:
            colnames = [f"enc_{i}" for i in range(getattr(Xt, 'shape', [0,0])[1])]
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        X_num = X.drop(columns=cat_cols)
        X_enc = pd.DataFrame(Xt, columns=colnames, index=X.index)
        X_final = pd.concat([X_num.reset_index(drop=True), X_enc.reset_index(drop=True)], axis=1)
        return X_final, cat_cols
    except Exception as e:
        st.warning(f"No se pudo aplicar el encoder: {e}")
        return X, []

def ensure_dataframe(input_dict):
    return pd.DataFrame([input_dict])

# =============================
# UI principal
# =============================
st.title("💳 Credit Risk Prediction")
st.write("Modelo de riesgo con categorías **housing**, **job**, **purpose** y soporte para CSV por lotes.")

# ------ Entrada manual
st.subheader("Entrada manual")
col1, col2, col3 = st.columns(3)
with col1:
    credit_amount = st.number_input("Credit Amount", min_value=0.0, step=100.0)
with col2:
    housing_rent = st.number_input("Housing Rent (mensual)", min_value=0.0, step=50.0)
with col3:
    age = st.number_input("Age", min_value=18, step=1, value=30)

st.markdown("#### Categorías")
housing_opts = ["own", "rent", "free", "other"]
job_opts = ["unskilled", "skilled", "management", "self-employed", "unemployed", "other"]
purpose_opts = ["car", "furniture/equipment", "education", "business", "domestic appliances", "repairs", "other"]

c1, c2, c3 = st.columns(3)
with c1:
    housing = st.selectbox("housing", housing_opts, index=1)
with c2:
    job = st.selectbox("job", job_opts, index=1)
with c3:
    purpose = st.selectbox("purpose", purpose_opts, index=len(purpose_opts)-1)

c1a, c2a, c3a = st.columns(3)
with c1a:
    housing_custom = st.text_input("Otro housing (opcional)", "")
with c2a:
    job_custom = st.text_input("Otro job (opcional)", "")
with c3a:
    purpose_custom = st.text_input("Otro purpose (opcional)", "")

housing_val = housing_custom.strip() if housing == "other" and housing_custom.strip() else housing
job_val = job_custom.strip() if job == "other" and job_custom.strip() else job
purpose_val = purpose_custom.strip() if purpose == "other" and purpose_custom.strip() else purpose

if st.button("🔮 Predecir (entrada manual)"):
    X = ensure_dataframe({
        "credit_amount": credit_amount,
        "housing_rent": housing_rent,
        "age": age,
        "housing": str(housing_val),
        "job": str(job_val),
        "purpose": str(purpose_val),
    })
    # (NUEVO) normaliza nombres hacia el esquema entrenado
    X = normalize_feature_names(X)

    X_proc, used_cats = apply_encoder(ENCODER_OBJ, X.copy())
    try:
        if hasattr(MODEL_OBJ, "predict_proba"):
            proba = MODEL_OBJ.predict_proba(X_proc)
            p_good = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0])
            y_classes = getattr(MODEL_OBJ, "classes_", np.array(["Bad","Good"]))
            label = y_classes[int(np.argmax(proba, axis=1)[0])] if proba.ndim == 2 else ("Good" if p_good >= 0.5 else "Bad")
            st.success(f"Predicción: **{label}** | Prob(💚Good): **{p_good:.3f}**")
            st.caption(f"Categorías usadas: {used_cats if used_cats else '—'}")
        elif hasattr(MODEL_OBJ, "predict"):
            pred_raw = MODEL_OBJ.predict(X_proc)
            st.success(f"Predicción: **{str(pred_raw[0])}**")
            st.caption(f"Categorías usadas: {used_cats if used_cats else '—'}")
        else:
            raise AttributeError("El objeto del modelo no tiene .predict ni .predict_proba")
    except Exception as e:
        st.warning(f"No se pudo usar el modelo guardado. Se usa *fallback*. Detalle: {e}")
        out = fallback_predict(X)
        st.info(f"Predicción (*fallback*): **{out.loc[0,'prediction']}** | Prob(💚Good): **{out.loc[0,'proba_good_fallback']:.3f}**")

# ------ CSV por lotes
st.subheader("📎 Lote por CSV")
st.write("Incluye columnas numéricas (`credit_amount`, `housing_rent`, `age`) y categóricas (`housing`, `job`, `purpose`).")
uploaded = st.file_uploader("Sube un CSV", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write("Vista previa:", df.head())
        for c in ["credit_amount", "housing_rent", "age", "housing", "job", "purpose"]:
            if c not in df.columns:
                df[c] = np.nan if c in ["housing","job","purpose"] else 0.0

        # (NUEVO) normaliza nombres hacia el esquema entrenado
        df = normalize_feature_names(df)

        X_proc, used_cats = apply_encoder(ENCODER_OBJ, df.copy())
        try:
            if hasattr(MODEL_OBJ, "predict_proba"):
                proba = MODEL_OBJ.predict_proba(X_proc)
                p_good = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
                pred = np.where(p_good >= 0.5, "Good", "Bad")
                res = df.copy()
                res["prob_good"] = p_good
                res["prediction"] = pred
            elif hasattr(MODEL_OBJ, "predict"):
                pred = MODEL_OBJ.predict(X_proc)
                res = df.copy()
                res["prediction"] = pred
            else:
                raise AttributeError("El objeto del modelo no tiene .predict ni .predict_proba")

            st.success("Predicciones generadas.")
            st.dataframe(res.head(50))
            st.download_button("⬇️ Descargar resultados CSV", data=res.to_csv(index=False).encode("utf-8"),
                               file_name="predicciones_credit_risk.csv", mime="text/csv")
        except Exception as me:
            st.warning(f"No se pudo usar el modelo guardado. Se usa *fallback*. Detalle: {me}")
            fb = fallback_predict(df)
            res = pd.concat([df.reset_index(drop=True), fb.reset_index(drop=True)], axis=1)
            st.dataframe(res.head(50))
            st.download_button("⬇️ Descargar resultados CSV (fallback)", data=res.to_csv(index=False).encode("utf-8"),
                               file_name="predicciones_credit_risk_fallback.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error leyendo el CSV: {e}")

st.markdown("---")
st.caption("Incluye categorías: housing, job, purpose. Carga artefactos desde rutas o súbelos en la barra lateral.")








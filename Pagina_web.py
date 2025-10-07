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
    """Regla simple (demo)."""
    credit_amount = df.get("credit_amount", pd.Series([0]*len(df))).astype(float)
    duration      = df.get("duration", pd.Series([12]*len(df))).astype(float)
    age           = df.get("age", pd.Series([30]*len(df))).astype(float)
    job_num       = pd.to_numeric(df.get("job", pd.Series([0]*len(df))), errors="coerce").fillna(0)

    housing = df.get("housing", pd.Series(["other"]*len(df))).astype(str).str.lower()
    purpose = df.get("purpose", pd.Series(["other"]*len(df))).astype(str).str.lower()

    cat_bonus = (
        housing.isin(["own"]).astype(int)*0.10
        + (job_num >= 2).astype(int)*0.08
        + purpose.isin(["education","furniture/equipment"]).astype(int)*0.05
    )

    score = (age/100.0) + (duration/48.0) - (credit_amount/10000.0) + cat_bonus
    proba = (score - score.min()) / (score.max() - score.min() + 1e-6)
    yhat  = np.where(score > 0, "Good", "Bad")
    return pd.DataFrame({"score_fallback": score, "proba_good_fallback": proba, "prediction": yhat})

# === MAPEOS SEGUROS (ACTUALIZADO) =====================================
RENAME_MAP_SAFE = {
    "housing": "Housing", "Housing": "Housing",

    "purpose": "Purpose", "Purpuso": "Purpose", "Purpose": "Purpose",

    "saving accounts": "Saving accounts", "Saving accounts": "Saving accounts",
    "Saving accounts ": "Saving accounts", "saving_accounts": "Saving accounts",

    "checking account": "Checking account", "Checking account": "Checking account",
    "checking_account": "Checking account",

    # numéricas / candidatas
    "age": "Age", "Age": "Age",
    "job": "Job", "Job": "Job",   # llega numérica desde la UI
    "credit amount": "Credit amount", "Credit amount": "Credit amount", "credit_amount": "Credit amount",
    "duration": "Duration", "Duration": "Duration",
    "housing_rent": "Housing_rent", "Housing_rent": "Housing_rent",
}

# ⚠️ Esquema REAL del entrenamiento:
#  - Job fue CATEGÓRICA (tiene dummies)
#  - Se usó Housing_rent (no Duration)
CAT_TRAIN_COLS = ["Checking account", "Saving accounts", "Housing", "Purpose", "Job"]
NUM_TRAIN_COLS = ["Age", "Credit amount", "Housing_rent"]

# Niveles válidos de Job que vio el encoder/modelo
JOB_ALLOWED = ["1.0", "1.2", "1.6", "2.0", "3.0"]

def _coerce_job_to_trained_levels(val) -> str:
    """Convierte job numérico a la categoría entrenada más cercana ('1.0','1.2','1.6','2.0','3.0')."""
    try:
        x = float(val)
    except Exception:
        return JOB_ALLOWED[0]
    levels = [float(v) for v in JOB_ALLOWED]
    nearest = min(levels, key=lambda v: abs(v - x))
    return f"{nearest:.1f}"

def normalize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ajusta nombres/columnas al esquema del entrenamiento sin tocar nada más del flujo."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME_MAP_SAFE)

    # 1) Duration (UI) -> Housing_rent (entrenamiento)
    if "Duration" in df.columns and "Housing_rent" not in df.columns:
        df["Housing_rent"] = pd.to_numeric(df["Duration"], errors="coerce").fillna(0.0)
        df.drop(columns=["Duration"], inplace=True)

    # 2) Job numérica (UI) -> categórica en niveles entrenados
    if "Job" in df.columns:
        df["Job"] = df["Job"].apply(_coerce_job_to_trained_levels).astype("category")

    # 3) Garantiza columnas esperadas y tipos
    for c in CAT_TRAIN_COLS:
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].astype("category")

    for c in NUM_TRAIN_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    # Orden prolijo (opcional, no afecta el modelo)
    ordered_cols = CAT_TRAIN_COLS + NUM_TRAIN_COLS
    df = df[ordered_cols + [c for c in df.columns if c not in ordered_cols]]
    return df
# =======================================================================

def apply_encoder(enc, X: pd.DataFrame):
    """Aplica OneHotEncoder si existe; concatena numéricas + codificadas."""
    if enc is None:
        return X, []
    try:
        X = normalize_feature_names(X)
        cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
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
st.write("Modelo de riesgo con **Housing**, **Saving accounts**, **Checking account**, **Purpose** (categóricas) y **Age**, **Job (numérico)**, **Credit amount**, **Duration** (numéricas).")

# ------ Entrada manual
st.subheader("Entrada manual")
col1, col2, col3 = st.columns(3)
with col1:
    credit_amount = st.number_input("Credit Amount", min_value=0.0, step=100.0)
with col2:
    duration = st.number_input("Duration", min_value=1, step=1, value=12)
with col3:
    age = st.number_input("Age", min_value=18, step=1, value=30)

st.markdown("#### Categorías")
housing_opts = ["own", "free", "rent"]
saving_accounts_opts = ["little", "quite rich", "moderate", "rich"]
checking_account_opts = ["little", "moderate", "rich"]
purpose_opts = ["business", "car", "domestic appliances", "education",
                "furniture/equipment", "radio/TV", "repairs", "vacation/others"]

c1, c2, c3 = st.columns(3)
with c1:
    housing = st.selectbox("housing", housing_opts, index=0)
with c2:
    saving_accounts = st.selectbox("Saving accounts", saving_accounts_opts, index=0)
with c3:
    checking_account = st.selectbox("Checking account", checking_account_opts, index=0)

c4, c5 = st.columns(2)
with c4:
    # Job numérica
    job_num = st.number_input("job (numérico)", min_value=0, max_value=10, value=2, step=1)
with c5:
    purpose = st.selectbox("purpose", purpose_opts, index=0)

if st.button("🔮 Predecir (entrada manual)"):
    X = ensure_dataframe({
        "credit_amount": credit_amount,
        "duration": duration,
        "age": age,
        "housing": str(housing),
        "Saving accounts": str(saving_accounts),   # ya con nombre entrenado
        "Checking account": str(checking_account), # ya con nombre entrenado
        "job": job_num,                             # numérico en UI
        "purpose": str(purpose),
    })
    # normaliza nombres hacia el esquema entrenado
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
st.write("Incluye columnas numéricas (`credit_amount`, `duration`, `age`, `job`) y categóricas (`housing`, `Saving accounts`, `Checking account`, `purpose`).")
uploaded = st.file_uploader("Sube un CSV", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write("Vista previa:", df.head())

        needed = ["credit_amount", "duration", "age", "job", "housing",
                  "Saving accounts", "Checking account", "purpose"]
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan if c in ["housing","Saving accounts","Checking account","purpose"] else 0.0

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
st.caption("UI alineada al entrenamiento: Checking account, Saving accounts, Housing, Purpose (cat) + Age, Job (num), Credit amount, Duration (num).")




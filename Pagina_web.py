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
KNN_CAT = None
KNN_NUM = None
LABEL_ENC = None

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
    model_path = st.sidebar.text_input("Ruta del modelo (.pkl)", "./xgboost_credit_risk_20251004_163147.pkl")
    enc_path   = st.sidebar.text_input("Ruta OneHotEncoder (opcional)", "./onehot_encoder_20251004_165417.pkl")
    wts_path   = st.sidebar.text_input("Ruta class weights (opcional)", "./balance_info_class_weights.pkl")

    knn_cat_path   = st.sidebar.text_input("Ruta KNN Imputer (categ.)", "./knn_imputer_categorical_20251004_165417.pkl")
    knn_num_path   = st.sidebar.text_input("Ruta KNN Imputer (num.)",    "./knn_imputer_numerical_20251004_165417.pkl")
    label_enc_path = st.sidebar.text_input("Ruta LabelEncoders (opc.)",  "./label_encoders_20251004_165417.pkl")

    def load_if_exists(path):
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return safe_unpickle(f)
        return (None, f"No encontrado: {path}")

    MODEL_OBJ, m_err = load_if_exists(model_path)
    ENCODER_OBJ, e_err = load_if_exists(enc_path) if enc_path else (None, None)
    CLASSWEIGHTS_OBJ, w_err = load_if_exists(wts_path) if wts_path else (None, None)

    KNN_CAT,  knc_err = load_if_exists(knn_cat_path)
    KNN_NUM,  knn_err = load_if_exists(knn_num_path)
    LABEL_ENC, le_err = load_if_exists(label_enc_path)
else:
    st.sidebar.caption("Sube tus artefactos entrenados:")
    up_model = st.sidebar.file_uploader("Modelo (.pkl)", type=["pkl"], key="mdl")
    up_enc   = st.sidebar.file_uploader("OneHotEncoder (.pkl, opcional)", type=["pkl"], key="enc")
    up_wts   = st.sidebar.file_uploader("Class Weights (.pkl, opcional)", type=["pkl"], key="wts")

    up_knn_cat = st.sidebar.file_uploader("KNN Imputer (categ.)", type=["pkl"], key="knncat")
    up_knn_num = st.sidebar.file_uploader("KNN Imputer (num.)",    type=["pkl"], key="knnnum")
    up_le      = st.sidebar.file_uploader("LabelEncoders (opc.)",  type=["pkl"], key="les")

    if up_model is not None:
        MODEL_OBJ, m_err = safe_unpickle(up_model)
    else:
        MODEL_OBJ, m_err = (None, "Falta el modelo")
    ENCODER_OBJ, e_err = safe_unpickle(up_enc) if up_enc is not None else (None, None)
    CLASSWEIGHTS_OBJ, w_err = safe_unpickle(up_wts) if up_wts is not None else (None, None)

    KNN_CAT,  knc_err = safe_unpickle(up_knn_cat) if up_knn_cat else (None, None)
    KNN_NUM,  knn_err = safe_unpickle(up_knn_num) if up_knn_num else (None, None)
    LABEL_ENC, le_err = safe_unpickle(up_le)      if up_le      else (None, None)

# Avisos de carga
for err in [m_err, e_err, w_err,
            (locals().get("knc_err", None)),
            (locals().get("knn_err", None)),
            (locals().get("le_err", None))]:
    if err:
        st.sidebar.warning(err)

# =============================
# Helpers
# =============================
def fallback_predict(df: pd.DataFrame):
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

RENAME_MAP_SAFE = {
    "housing": "Housing", "Housing": "Housing",
    "purpose": "Purpose", "Purpose": "Purpose",
    "saving accounts": "Saving accounts", "Saving accounts": "Saving accounts",
    "Saving accounts ": "Saving accounts", "saving_accounts": "Saving accounts",
    "checking account": "Checking account", "Checking account": "Checking account",
    "checking_account": "Checking account",
    "age": "Age", "Age": "Age",
    "job": "Job", "Job": "Job",
    "credit amount": "Credit amount", "Credit amount": "Credit amount", "credit_amount": "Credit amount",
    "duration": "Duration", "Duration": "Duration",
}

CAT_TRAIN_COLS = ["Checking account", "Saving accounts", "Housing", "Purpose", "Job"]
NUM_TRAIN_COLS = ["Age", "Credit amount", "Duration"]
JOB_ALLOWED = ["1.0", "1.2", "1.6", "2.0", "3.0"]

def _coerce_job_to_trained_levels(val) -> str:
    try:
        x = float(val)
    except Exception:
        return JOB_ALLOWED[0]
    levels = [float(v) for v in JOB_ALLOWED]
    nearest = min(levels, key=lambda v: abs(v - x))
    return f"{nearest:.1f}"

def normalize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME_MAP_SAFE)
    if "Job" in df.columns:
        df["Job"] = df["Job"].apply(_coerce_job_to_trained_levels).astype("category")
    for c in CAT_TRAIN_COLS:
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].astype("category")
    for c in NUM_TRAIN_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    ordered_cols = CAT_TRAIN_COLS + NUM_TRAIN_COLS
    return df[ordered_cols + [c for c in df.columns if c not in ordered_cols]]

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
    job_num = st.number_input("job (numérico)", min_value=0, max_value=10, value=2, step=1)
with c5:
    purpose = st.selectbox("purpose", purpose_opts, index=0)

def apply_saved_imputers(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica KNN imputers (num y cat) con los LabelEncoders guardados."""
    df = df.copy()

    # Numéricas
    if KNN_NUM is not None:
        num_cols = list(getattr(KNN_NUM, "feature_names_in_", df.select_dtypes(include="number").columns))
        for c in num_cols:
            if c not in df.columns: df[c] = np.nan
        num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
        df[num_cols] = pd.DataFrame(KNN_NUM.transform(num_df), columns=num_cols, index=df.index)

    # Categóricas
    if KNN_CAT is not None and isinstance(LABEL_ENC, dict):
        cat_cols = list(getattr(KNN_CAT, "feature_names_in_", LABEL_ENC.keys()))
        for c in cat_cols:
            if c not in df.columns: df[c] = np.nan
        enc_df = pd.DataFrame(index=df.index)
        for c in cat_cols:
            le = LABEL_ENC.get(c)
            enc_df[c] = encode_for_knn_cat(df[c].astype(str), le) if le else np.nan
        enc_imputed = pd.DataFrame(KNN_CAT.transform(enc_df), columns=cat_cols, index=df.index)
        for c in cat_cols:
            le = LABEL_ENC.get(c)
            if le: df[c] = inverse_clip(enc_imputed[c], le)
    return df

def apply_encoder(enc, X: pd.DataFrame):
    """OneHot + concat con numéricas"""
    if enc is None: return X, []
    X = normalize_feature_names(X)
    X = apply_saved_imputers(X)

    if hasattr(enc, "feature_names_in_"):
        req = list(enc.feature_names_in_)
    else:
        req = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]

    for c in req:
        if c not in X.columns: X[c] = "__missing__"
        X[c] = X[c].astype(str).fillna("__missing__")

    Z = enc.transform(X[req])
    if hasattr(Z, "toarray"): Z = Z.toarray()
    try:
        names = enc.get_feature_names_out(req)
    except Exception:
        names = [f"enc_{i}" for i in range(Z.shape[1])]
    X_enc = pd.DataFrame(Z, columns=names, index=X.index)
    X_num = X.drop(columns=req, errors="ignore").select_dtypes(include=["number"])
    return pd.concat([X_num, X_enc], axis=1), req

def ensure_dataframe(input_dict):
    return pd.DataFrame([input_dict])

if st.button("🔮 Predecir (entrada manual)"):
    X = ensure_dataframe({
        "credit_amount": credit_amount,
        "duration": duration,
        "age": age,
        "housing": str(housing),
        "Saving accounts": str(saving_accounts),
        "Checking account": str(checking_account),
        "job": job_num,
        "purpose": str(purpose),
    })
    X = normalize_feature_names(X)
    X_proc, used_cats = apply_encoder(ENCODER_OBJ, X.copy())
    try:
        if hasattr(MODEL_OBJ, "predict_proba"):
            proba = MODEL_OBJ.predict_proba(align_to_model_features(X_proc, MODEL_OBJ))
            p_good = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0])
            y_classes = getattr(MODEL_OBJ, "classes_", np.array(["Bad","Good"]))
            label = y_classes[int(np.argmax(proba, axis=1)[0])] if proba.ndim == 2 else ("Good" if p_good >= 0.5 else "Bad")
            st.success(f"Predicción: **{label}** | Prob(💚Good): **{p_good:.3f}**")
            st.caption(f"Categorías usadas (orden OHE): {used_cats if used_cats else '—'}")
        else:
            pred_raw = MODEL_OBJ.predict(align_to_model_features(X_proc, MODEL_OBJ))
            st.success(f"Predicción: **{str(pred_raw[0])}**")
            st.caption(f"Categorías usadas (orden OHE): {used_cats if used_cats else '—'}")
    except Exception as e:
        st.warning(f"No se pudo usar el modelo guardado. Se usa fallback. Detalle: {e}")
        out = fallback_predict(X)
        st.info(f"Predicción (*fallback*): **{out.loc[0,'prediction']}** | Prob(💚Good): **{out.loc[0,'proba_good_fallback']:.3f}**")

# ------ CSV por lotes (misma lógica del snippet que te funciona)
st.subheader("📎 Lote por CSV")
st.write("Incluye columnas numéricas (`credit_amount`, `duration`, `age`, `job`) y categóricas (`housing`, `Saving accounts`, `Checking account`, `purpose`).")

file_csv = st.file_uploader("CSV", type=["csv"], key="csv_batch")

if file_csv is not None:
    try:
        raw = pd.read_csv(file_csv)
        st.write("Vista previa:")
        st.dataframe(raw.head())
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        raw = None

    if raw is not None:
        df = normalize_feature_names(raw.copy())
        for c in df.columns:
            df[c] = soft_to_numeric(df[c])

        # Imputación categóricas
        if KNN_CAT is not None and isinstance(LABEL_ENC, dict):
            cat_cols = list(getattr(KNN_CAT, "feature_names_in_", LABEL_ENC.keys()))
            for c in cat_cols:
                if c not in df.columns: df[c] = np.nan
            enc_df = pd.DataFrame(index=df.index)
            for c in cat_cols:
                le = LABEL_ENC.get(c)
                enc_df[c] = encode_for_knn_cat(df[c].astype(str), le) if le else np.nan
            enc_imputed = pd.DataFrame(KNN_CAT.transform(enc_df), columns=cat_cols, index=df.index)
            for c in cat_cols:
                le = LABEL_ENC.get(c)
                if le: df[c] = inverse_clip(enc_imputed[c], le)
            st.info("Imputación KNN (categóricas) ok.")

        # Imputación numéricas
        if KNN_NUM is not None:
            num_cols = list(getattr(KNN_NUM, "feature_names_in_", df.select_dtypes(include="number").columns))
            for c in num_cols:
                if c not in df.columns: df[c] = np.nan
            num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
            num_imputed = pd.DataFrame(KNN_NUM.transform(num_df), columns=num_cols, index=df.index)
            df[num_cols] = num_imputed
            st.info("Imputación KNN (numéricas) ok.")

        # OneHot
        if ENCODER_OBJ is None:
            st.error("Falta OneHotEncoder.")
        else:
            ohe_in_cols = list(getattr(ENCODER_OBJ, "feature_names_in_", []))
            if not ohe_in_cols:
                ohe_in_cols = [c for c in df.columns if df[c].dtype == "object"]
            for c in ohe_in_cols:
                if c not in df.columns: df[c] = "__missing__"
                df[c] = df[c].fillna("__missing__").astype(str)

            Z_cat = ENCODER_OBJ.transform(df[ohe_in_cols])
            if hasattr(Z_cat, "toarray"): Z_cat = Z_cat.toarray()
            try:
                cat_out_cols = ENCODER_OBJ.get_feature_names_out(ohe_in_cols)
            except Exception:
                cat_out_cols = [f"cat_{i}" for i in range(Z_cat.shape[1])]
            df_cat = pd.DataFrame(Z_cat, columns=cat_out_cols, index=df.index)

            # Numéricas finales
            X_num = df.drop(columns=ohe_in_cols, errors="ignore").select_dtypes(include=["number"])
            X = pd.concat([X_num, df_cat], axis=1)
            X = align_to_model_features(X, MODEL_OBJ)

            if st.button("Predict", key="predict_csv"):
                try:
                    y = MODEL_OBJ.predict(X)
                    mapping = {0: "Good", 1: "Bad"}
                    y_pretty = [mapping.get(int(v), str(v)) for v in y]
                    out = raw.copy()
                    out["Prediction"] = y_pretty
                    st.subheader("Resultados")
                    st.dataframe(out.head(50))
                    st.download_button(
                        "Descargar (CSV)",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="predicciones_risk.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error durante la predicción: {e}")
                    st.write("Cols procesadas:", list(X.columns))
                    if hasattr(MODEL_OBJ, "feature_names_in_"):
                        st.write("Cols esperadas por el modelo:", list(MODEL_OBJ.feature_names_in_))

st.markdown("---")
st.caption("Diseñado,Montado y desplegado por Sebastian Rugeles Sanchez")


























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
KNN_CAT = None            # ← NUEVO
KNN_NUM = None            # ← NUEVO
LABEL_ENC = None          # ← NUEVO

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

    # ← NUEVO: rutas para imputers y label encoders
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

    # ← NUEVO: carga de imputers y label encoders
    KNN_CAT,  knc_err = load_if_exists(knn_cat_path)
    KNN_NUM,  knn_err = load_if_exists(knn_num_path)
    LABEL_ENC, le_err = load_if_exists(label_enc_path)
else:
    st.sidebar.caption("Sube tus artefactos entrenados:")
    up_model = st.sidebar.file_uploader("Modelo (.pkl)", type=["pkl"], key="mdl")
    up_enc   = st.sidebar.file_uploader("OneHotEncoder (.pkl, opcional)", type=["pkl"], key="enc")
    up_wts   = st.sidebar.file_uploader("Class Weights (.pkl, opcional)", type=["pkl"], key="wts")

    # ← NUEVO: uploads para imputers y label encoders
    up_knn_cat = st.sidebar.file_uploader("KNN Imputer (categ.)", type=["pkl"], key="knncat")
    up_knn_num = st.sidebar.file_uploader("KNN Imputer (num.)",    type=["pkl"], key="knnnum")
    up_le      = st.sidebar.file_uploader("LabelEncoders (opc.)",  type=["pkl"], key="les")

    if up_model is not None:
        MODEL_OBJ, m_err = safe_unpickle(up_model)
    else:
        MODEL_OBJ, m_err = (None, "Falta el modelo")
    ENCODER_OBJ, e_err = safe_unpickle(up_enc) if up_enc is not None else (None, None)
    CLASSWEIGHTS_OBJ, w_err = safe_unpickle(up_wts) if up_wts is not None else (None, None)

    # ← NUEVO: carga desde uploads
    KNN_CAT,  knc_err = safe_unpickle(up_knn_cat) if up_knn_cat else (None, None)
    KNN_NUM,  knn_err = safe_unpickle(up_knn_num) if up_knn_num else (None, None)
    LABEL_ENC, le_err = safe_unpickle(up_le)      if up_le      else (None, None)

# Avisos de carga
for err in [m_err, e_err, w_err]:
    if err:
        st.sidebar.warning(err)
# ← NUEVO: avisos imputers/label enc
for err in [knc_err if 'knc_err' in locals() else None,
            knn_err if 'knn_err' in locals() else None,
            le_err  if 'le_err'  in locals() else None]:
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
}

# Esquema real del entrenamiento:
#  - Job categórica (dummies)
#  - Duration numérica
CAT_TRAIN_COLS = ["Checking account", "Saving accounts", "Housing", "Purpose", "Job"]
NUM_TRAIN_COLS = ["Age", "Credit amount", "Duration"]

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

    # Job numérica (UI) -> categórica en niveles entrenados
    if "Job" in df.columns:
        df["Job"] = df["Job"].apply(_coerce_job_to_trained_levels).astype("category")

    # Garantiza columnas esperadas y tipos
    for c in CAT_TRAIN_COLS:
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].astype("category")

    for c in NUM_TRAIN_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    # Orden prolijo (opcional)
    ordered_cols = CAT_TRAIN_COLS + NUM_TRAIN_COLS
    df = df[ordered_cols + [c for c in df.columns if c not in ordered_cols]]
    return df
# =======================================================================

# ← NUEVO: utilidades para aplicar imputers entrenados
def _safe_label_transform(le, val):
    try:
        if pd.isna(val):
            return np.nan
        if str(val) in set(map(str, le.classes_)):
            return le.transform([str(val)])[0]
        return np.nan
    except Exception:
        return np.nan

def apply_saved_imputers(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica KNN imputers (num y cat) y label encoders entrenados.
       Debe llamarse DESPUÉS de normalize_feature_names y ANTES del OneHotEncoder."""
    df = df.copy()

    # ===== NUMÉRICAS =====
    num_base = [c for c in ["Age", "Credit amount", "Duration"] if c in df.columns]
    if KNN_NUM is not None and num_base:
        # orden/nombres EXACTOS del imputer si existen
        if hasattr(KNN_NUM, "feature_names_in_"):
            num_order = [str(c) for c in KNN_NUM.feature_names_in_]
        else:
            num_order = num_base

        # asegurar existencia y tipo numérico
        for c in num_order:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")

        Xnum = df[num_order]
        imputed_num = KNN_NUM.transform(Xnum)
        df[num_order] = imputed_num

    # ===== CATEGÓRICAS =====
    cat_base = [c for c in ["Saving accounts", "Checking account", "Housing", "Purpose", "Job"] if c in df.columns]
    if KNN_CAT is not None and LABEL_ENC is not None and cat_base:
        # orden/nombres EXACTOS del imputer si existen
        if hasattr(KNN_CAT, "feature_names_in_"):
            cat_order = [str(c) for c in KNN_CAT.feature_names_in_]
        else:
            cat_order = cat_base

        # construir matriz codificada con el MISMO orden
        enc_mat = pd.DataFrame(index=df.index)
        for c in cat_order:
            le = LABEL_ENC.get(c) if isinstance(LABEL_ENC, dict) else None
            if le is None:
                enc_mat[c] = np.nan
            else:
                enc_mat[c] = df[c].astype(object).apply(lambda x: _safe_label_transform(le, x))

        enc_mat = enc_mat[cat_order]

        imputed_cat = KNN_CAT.transform(enc_mat)
        imputed_df = pd.DataFrame(np.rint(imputed_cat).astype(int), columns=cat_order, index=df.index)

        # inverse_transform con vector 1-D y rango válido
        for c in cat_order:
            le = LABEL_ENC.get(c) if isinstance(LABEL_ENC, dict) else None
            if le is not None:
                arr = np.clip(imputed_df[c].astype(int).to_numpy(), 0, len(le.classes_) - 1)
                df[c] = le.inverse_transform(arr)

    return df

def apply_encoder(enc, X: pd.DataFrame):
    """Aplica OneHotEncoder si existe; concatena numéricas + codificadas."""
    if enc is None:
        return X, []
    try:
        # Normaliza nombres y tipos
        X = normalize_feature_names(X)
        # ← NUEVO: aplica imputers entrenados ANTES del OHE
        X = apply_saved_imputers(X)

        # Usa EXACTAMENTE el orden y conjunto del encoder
        if hasattr(enc, "feature_names_in_"):
            req = list(enc.feature_names_in_)  # columnas categóricas con las que se entrenó el OHE
        else:
            # fallback: detectar por dtype (no ideal)
            req = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]

        # Asegura que todas existan
        for c in req:
            if c not in X.columns:
                X[c] = pd.Categorical([None]*len(X))

        df_cat = X[req].copy()
        for c in df_cat.columns:
            df_cat[c] = df_cat[c].astype("category")

        # Transforma en el orden correcto
        Xt = enc.transform(df_cat)
        try:
            colnames = enc.get_feature_names_out(req)
        except Exception:
            colnames = [f"enc_{i}" for i in range(getattr(Xt, 'shape', [0,0])[1])]

        # 👉 Garantiza matriz 2-D para construir DataFrame
        if hasattr(Xt, "toarray"):   # sparse -> dense
            Xt = Xt.toarray()
        Xt = np.asarray(Xt)
        if Xt.ndim == 1:
            Xt = Xt.reshape(-1, 1)

        # Separamos numéricas (todo lo que no es categórica original del OHE)
        X_num = X.drop(columns=req, errors="ignore")
        X_enc = pd.DataFrame(Xt, columns=colnames, index=X.index)
        X_final = pd.concat([X_num.reset_index(drop=True), X_enc.reset_index(drop=True)], axis=1)
        return X_final, req
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
        "Saving accounts": str(saving_accounts),
        "Checking account": str(checking_account),
        "job": job_num,           # numérico en UI → luego convertimos a categoría entrenada
        "purpose": str(purpose),
    })
    # normaliza nombres hacia el esquema entrenado
    X = normalize_feature_names(X)
    # ← NUEVO: aplica imputers guardados
    X = apply_saved_imputers(X)

    X_proc, used_cats = apply_encoder(ENCODER_OBJ, X.copy())
    try:
        if hasattr(MODEL_OBJ, "predict_proba"):
            proba = MODEL_OBJ.predict_proba(X_proc)
            p_good = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0])
            y_classes = getattr(MODEL_OBJ, "classes_", np.array(["Bad","Good"]))
            label = y_classes[int(np.argmax(proba, axis=1)[0])] if proba.ndim == 2 else ("Good" if p_good >= 0.5 else "Bad")
            st.success(f"Predicción: **{label}** | Prob(💚Good): **{p_good:.3f}**")
            st.caption(f"Categorías usadas (orden OHE): {used_cats if used_cats else '—'}")
        elif hasattr(MODEL_OBJ, "predict"):
            pred_raw = MODEL_OBJ.predict(X_proc)
            st.success(f"Predicción: **{str(pred_raw[0])}**")
            st.caption(f"Categorías usadas (orden OHE): {used_cats if used_cats else '—'}")
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
        # 👉 maneja miles con coma y convierte 'None' / vacíos a NaN
        df = pd.read_csv(
            uploaded,
            thousands=",",
            keep_default_na=True,
            na_values=["None", "NA", "NaN", ""]
        )
        st.write("Vista previa:", df.head())

        # ⬇️ NUEVO: limpia miles/espacios por si vinieron como string
        if "Credit amount" in df.columns:
            df["Credit amount"] = (
                df["Credit amount"].astype(str)
                                     .str.replace(r"[,\s]", "", regex=True)
                                     .replace("", np.nan)
            )

        needed = ["credit_amount", "duration", "age", "job", "housing",
                  "Saving accounts", "Checking account", "purpose"]
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan if c in ["housing","Saving accounts","Checking account","purpose"] else 0.0

        # Normaliza nombres/tipos al esquema del entrenamiento
        df = normalize_feature_names(df)
        # ← NUEVO: aplica imputers guardados
        df = apply_saved_imputers(df)

        # Predicción
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
            st.download_button(
                "⬇️ Descargar resultados CSV",
                data=res.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_credit_risk.csv",
                mime="text/csv"
            )
        except Exception as me:
            st.warning(f"No se pudo usar el modelo guardado. Se usa *fallback*. Detalle: {me}")
            fb = fallback_predict(df)
            res = pd.concat([df.reset_index(drop=True), fb.reset_index(drop=True)], axis=1)
            st.dataframe(res.head(50))
            st.download_button(
                "⬇️ Descargar resultados CSV (fallback)",
                data=res.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_credit_risk_fallback.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error procesando el CSV: {e}")

st.markdown("---")
st.caption("UI alineada al entrenamiento. OHE recibe exactamente feature_names_in_ y Job se mapea a niveles entrenados. CSV con thousands=',' y NaN para vacíos.")



















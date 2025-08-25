
import os
import re
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# Optional XGBoost import with a friendly message
try:
    import xgboost as xgb
    XGB_OK = True
except Exception as e:
    XGB_OK = False

st.set_page_config(page_title="Parkinson's via Voice — PD Probability", layout="wide")

st.title("🧑‍⚕️ Parkinson's Detection via Voice Biomarkers")
st.markdown("""
This app trains **SVM (RBF)** and **XGBoost** models on the UCI Parkinson's Voice dataset (195 recordings, 22 features)
and lets you estimate **PD probability** for new inputs. It also shows **ROC curves**, **confusion matrices**, and **top biomarkers** by permutation importance.
""")

# -------------------------
# Data loading (robust)
# -------------------------
COLUMNS = [
    "name","MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)",
    "MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP",
    "MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5",
    "MDVP:APQ","Shimmer:DDA","NHR","HNR",
    "status","RPDE","DFA","spread1","spread2","D2","PPE"
]

@st.cache_data
def load_dataset(uploaded_file=None, path_hint=r"C:\Users\venka\Desktop\pdv\ds\parkinsons.data"):
    if uploaded_file is not None:
        raw = uploaded_file.getvalue().decode("utf-8")
        first = raw.splitlines()[0].strip()
        sep = "," if "," in first else (";" if ";" in first else None)
        if first.lower().startswith("name,"):
            df = pd.read_csv(io.StringIO(raw), sep=sep or ",", header=0)
        else:
            df = pd.read_csv(io.StringIO(raw), sep=sep or ",", header=None, names=COLUMNS)
    else:
        # try local file
        if not os.path.exists(path_hint):
            st.warning("Couldn't find 'parkinsons.data' next to the app. Please upload it in the sidebar.")
            return None
        with open(path_hint, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        sep = "," if "," in first else (";" if ";" in first else None)
        if first.lower().startswith("name,"):
            df = pd.read_csv(path_hint, sep=sep or ",", header=0)
        else:
            df = pd.read_csv(path_hint, sep=sep or ",", header=None, names=COLUMNS)
    # Ensure numeric dtypes
    for c in COLUMNS:
        if c != "name":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["status"]).dropna(axis=0)  # simple clean
    return df

def extract_subject_id(name: str) -> str:
    s = str(name)
    m = re.search(r"R(\d{1,2})", s, flags=re.IGNORECASE)
    if m:
        return f"R{int(m.group(1))}"
    parts = s.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2]).upper()
    return s.upper()

@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    if df is None or len(df) == 0:
        st.error("❌ Dataset is empty. Please upload a valid `parkinsons.data` file.")
        return None
    
    df = df.copy()
    df["subject_id"] = df["name"].astype(str).apply(extract_subject_id)

    X = df.drop(columns=["name","status","subject_id"])
    y = df["status"].astype(int)
    groups = df["subject_id"].astype(str)

    if len(X) == 0 or len(y) == 0:
        st.error("❌ No usable samples found after preprocessing.")
        return None

    # Choose split: group-safe if multiple subjects exist
    use_groups = groups.nunique() > 1

    if use_groups:
        splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        tr_idx, te_idx = next(splitter.split(X, y, groups))
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)

    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    # SVM pipeline
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", C=10, gamma=0.1))
    ])
    svm_pipe.fit(X_tr, y_tr)

    svm_proba_te = svm_pipe.predict_proba(X_te)[:,1]
    svm_pred_te  = (svm_proba_te >= 0.5).astype(int)
    svm_auc = roc_auc_score(y_te, svm_proba_te)
    svm_acc = accuracy_score(y_te, svm_pred_te)
    svm_cm  = confusion_matrix(y_te, svm_pred_te)

    # XGBoost (if available)
    xgb_model = None
    xgb_auc = xgb_acc = None
    xgb_proba_te = xgb_pred_te = None
    xgb_cm = None

    if XGB_OK:
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        spw = max(1.0, neg / max(1, pos))

        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=500,
            tree_method="hist",
            learning_rate=0.03,
            subsample=1.0,
            colsample_bytree=0.7,
            max_depth=4,
            min_child_weight=1,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            random_state=42
        )
        xgb_model.fit(X_tr, y_tr)
        xgb_proba_te = xgb_model.predict_proba(X_te)[:,1]
        xgb_pred_te  = (xgb_proba_te >= 0.5).astype(int)
        xgb_auc = roc_auc_score(y_te, xgb_proba_te)
        xgb_acc = accuracy_score(y_te, xgb_pred_te)
        xgb_cm  = confusion_matrix(y_te, xgb_pred_te)

    return {
        "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "svm": svm_pipe, "svm_auc": svm_auc, "svm_acc": svm_acc, "svm_cm": svm_cm,
        "xgb": xgb_model, "xgb_auc": xgb_auc, "xgb_acc": xgb_acc, "xgb_cm": xgb_cm,
        "use_groups": use_groups
    }

def plot_roc(y_true, proba, label):
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)

def plot_cm(cm, title):
    fig = plt.figure(figsize=(4,3))
    plt.imshow(cm, aspect="auto")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0,1], ["Healthy","PD"])
    plt.yticks([0,1], ["Healthy","PD"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    st.pyplot(fig)

def permutation_topk(model, X, y, k=10, title="Permutation Importance"):
    r = permutation_importance(model, X, y, n_repeats=20, random_state=0, scoring="roc_auc")
    imp = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean}).sort_values("importance", ascending=False).head(k)
    fig = plt.figure(figsize=(7,5))
    plt.barh(imp["feature"][::-1], imp["importance"][::-1])
    plt.title(title)
    plt.xlabel("AUC drop")
    st.pyplot(fig)

# -------------------------
# Sidebar: dataset upload
# -------------------------
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload parkinsons.data (optional)", type=["data","csv","txt"])
df = load_dataset(uploaded)

if df is None:
    st.stop()

st.success(f"Loaded dataset with shape {df.shape}.")

# -------------------------
# Train models
# -------------------------
with st.spinner("Training models..."):
    result = train_models(df)

X_tr = result["X_tr"]; X_te = result["X_te"]
y_tr = result["y_tr"]; y_te = result["y_te"]
svm = result["svm"]; xgb_model = result["xgb"]

st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("SVM AUC", f"{result['svm_auc']:.3f}")
    st.metric("SVM ACC", f"{result['svm_acc']:.3f}")
with col2:
    if xgb_model is not None:
        st.metric("XGB AUC", f"{result['xgb_auc']:.3f}")
        st.metric("XGB ACC", f"{result['xgb_acc']:.3f}")
    else:
        st.info("XGBoost not installed; only SVM is used.")
with col3:
    st.metric("Group-aware split", "Yes" if result["use_groups"] else "No")

# ROC curves
st.subheader("ROC Curves (Hold-out Test)")
colA, colB = st.columns(2)
with colA:
    svm_proba_te = svm.predict_proba(X_te)[:,1]
    plot_roc(y_te, svm_proba_te, "SVM")
with colB:
    if xgb_model is not None:
        xgb_proba_te = xgb_model.predict_proba(X_te)[:,1]
        plot_roc(y_te, xgb_proba_te, "XGBoost")

# Confusion matrices
st.subheader("Confusion Matrices")
colC, colD = st.columns(2)
with colC:
    svm_pred_te = (svm_proba_te >= 0.5).astype(int)
    plot_cm(result["svm_cm"], "SVM")
with colD:
    if xgb_model is not None:
        xgb_pred_te = (xgb_proba_te >= 0.5).astype(int)
        plot_cm(result["xgb_cm"], "XGBoost")

# -------------------------
# PD Probability — Single Input
# -------------------------
st.subheader("Predict PD Probability — Single Sample")

feature_defaults = df.drop(columns=["name","status"]).median(numeric_only=True).to_dict()
inputs = {}

with st.form("input_form"):
    cols = st.columns(3)
    ordered_cols = [c for c in X_tr.columns]
    for idx, colname in enumerate(ordered_cols):
        with cols[idx % 3]:
            val = float(feature_defaults.get(colname, 0.0))
            inputs[colname] = st.number_input(colname, value=val, format="%.6f")
    model_choice = st.selectbox("Model", ["XGBoost" if xgb_model is not None else "SVM", "SVM"] if xgb_model is not None else ["SVM"])
    submitted = st.form_submit_button("Predict")

if submitted:
    x_row = pd.DataFrame([inputs], columns=X_tr.columns)
    if model_choice == "XGBoost" and xgb_model is not None:
        proba = float(xgb_model.predict_proba(x_row)[0,1])
        st.success(f"PD Probability (XGBoost): **{proba:.3f}**")
    else:
        proba = float(svm.predict_proba(x_row)[0,1])
        st.success(f"PD Probability (SVM): **{proba:.3f}**")

# -------------------------
# PD Probability — Batch CSV
# -------------------------
st.subheader("Batch Prediction from CSV")
st.markdown("Upload a CSV with exactly these columns (22 features, no `name/status/subject_id`):")
st.code(", ".join(list(X_tr.columns)), language="text")
batch_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="batch")
if batch_file is not None:
    try:
        batch_df = pd.read_csv(batch_file)
        missing = [c for c in X_tr.columns if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            model_choice2 = st.selectbox("Batch model", ["XGBoost" if xgb_model is not None else "SVM", "SVM"], key="batch_model")
            if model_choice2 == "XGBoost" and xgb_model is not None:
                batch_proba = xgb_model.predict_proba(batch_df[X_tr.columns])[:,1]
            else:
                batch_proba = svm.predict_proba(batch_df[X_tr.columns])[:,1]
            out = batch_df.copy()
            out["PD_probability"] = batch_proba
            st.dataframe(out.head(50))
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv, file_name="pd_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")

# -------------------------
# Biomarker Importance
# -------------------------
st.subheader("Top Biomarkers (Permutation Importance on Test Split)")
colE, colF = st.columns(2)
with colE:
    permutation_topk(svm, X_te, y_te, k=10, title="SVM — Top 10 Biomarkers")
with colF:
    if xgb_model is not None:
        permutation_topk(xgb_model, X_te, y_te, k=10, title="XGBoost — Top 10 Biomarkers")

st.caption("Note: For rigorous evaluation, prefer group-aware splits (patients don't leak across train/test). This app auto-detects and uses group-aware splitting if subject IDs are available.")

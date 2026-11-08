from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import label_binarize

# =====================================================================
# CONFIG - edit this block only
# =====================================================================
STUDENT_NAME = "Your Name"
STUDENT_ID = "Your BITS ID"

TARGET = "Class"
DATASET_FACTS = {
    "Name": "Dry Bean Dataset (UCI id 602)",
    "Instances": "13,611",
    "Features": "16 numeric",
    "Classes": "7 bean varieties",
    "Train / test split": "10,888 / 2,723 (stratified, seed 42)",
    "Missing values": "0 (68 duplicate rows kept)",
}
CLASS_COUNTS = {"DERMASON": 3546, "SIRA": 2636, "SEKER": 2027, "HOROZ": 1928,
                "CALI": 1630, "BARBUNYA": 1322, "BOMBAY": 522}

# model name -> saved file (names match the training notebook)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "SVM": "svm.joblib",
}
MODEL_BLURB = {
    "Logistic Regression": "Linear model; estimates class probabilities from a weighted sum of scaled features.",
    "Decision Tree": "Rule-based model that splits the data on the most informative feature at each step.",
    "KNN": "Labels a bean by majority vote of its closest training beans (needs scaled features).",
    "Gaussian Naive Bayes": "Probabilistic model that assumes features are independent and bell-shaped within each class.",
    "SVM": "Finds maximum-margin boundaries between classes using an RBF kernel.",
}
FEATURE_INFO = {
    "Area": "Pixels inside the bean boundary",
    "Perimeter": "Length of the bean's border",
    "MajorAxisLength": "Longest line that can be drawn across the bean",
    "MinorAxisLength": "Longest line perpendicular to the major axis",
    "AspectRatio": "Major axis divided by minor axis",
    "Eccentricity": "Eccentricity of the ellipse with the same moments",
    "ConvexArea": "Pixels in the smallest convex polygon around the bean",
    "EquivDiameter": "Diameter of a circle with the same area",
    "Extent": "Bean area relative to its bounding box",
    "Solidity": "Bean area relative to its convex hull",
    "Roundness": "4 x pi x Area / Perimeter squared",
    "Compactness": "Equivalent diameter divided by major axis length",
    "ShapeFactor1": "Shape descriptor 1 (from the source paper)",
    "ShapeFactor2": "Shape descriptor 2 (from the source paper)",
    "ShapeFactor3": "Shape descriptor 3 (from the source paper)",
    "ShapeFactor4": "Shape descriptor 4 (from the source paper)",
}
SETTING_KEYS = ["C", "kernel", "gamma", "max_depth", "min_samples_leaf",
                "n_neighbors", "weights", "var_smoothing"]

ROOT = Path(__file__).resolve().parent
MODEL_DIR = next((p for p in (ROOT / "model", ROOT / "models") if p.is_dir()), None)
BUNDLED_TEST = ROOT / "test_data.csv"

# =====================================================================
# PAGE + STYLE
# =====================================================================
st.set_page_config(page_title="Dry Bean Variety Inspector", page_icon="🫘", layout="wide")
st.markdown("""
<style>
.hero {background: linear-gradient(120deg, #7b4a12 0%, #2f6b3a 100%); padding: 26px 32px;
       border-radius: 14px; color: #ffffff; margin-bottom: 18px;}
.hero h1 {margin: 0 0 6px 0; font-size: 2.0rem; color: #ffffff;}
.hero p {margin: 0; font-size: 1.05rem; opacity: 0.93; color: #ffffff;}
.pill {display: inline-block; padding: 3px 12px; margin: 12px 8px 0 0; border-radius: 999px;
       background: rgba(255,255,255,0.20); font-size: 0.82rem; color: #ffffff;}
.card {background-color: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.28);
       padding: 16px; border-radius: 10px; text-align: center;}
.card h4 {margin: 0 0 4px 0;}
.card hr {margin: 8px 0; border-color: rgba(128,128,128,0.3);}
.fact {display:flex; justify-content:space-between; padding:7px 2px;
       border-bottom:1px solid rgba(128,128,128,0.2); font-size:0.95rem;}
.fact span:first-child {opacity:0.7;}
.winner {border-left: 5px solid #2f6b3a; background: rgba(47,107,58,0.12);
         padding: 12px 16px; border-radius: 6px; margin: 8px 0 14px 0;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# HELPERS
# =====================================================================
@st.cache_resource(show_spinner="Loading saved models...")
def load_pipeline(path):
    return joblib.load(path)


def fmt(v):
    return "N/A" if pd.isna(v) else f"{v:.4f}"


def describe_settings(pipe):
    clf = pipe.steps[-1][1]
    params = clf.get_params()
    shown = {k: params[k] for k in SETTING_KEYS if k in params}
    steps = " > ".join(name for name, _ in pipe.steps[:-1]) or "none"
    return steps, ", ".join(f"{k}={v}" for k, v in shown.items())


@st.cache_data(show_spinner="Scoring the models...")
def score_model(model_name, data):
    """Predictions, probabilities and the six assignment metrics for one saved model."""
    pipe = load_pipeline(str(MODEL_DIR / MODEL_FILES[model_name]))
    X = data[list(pipe.feature_names_in_)]
    y = data[TARGET].astype(str)
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=pipe.classes_)
    except ValueError:
        auc = float("nan")
    scores = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return pred, proba, scores


# =====================================================================
# HEADER + SIDEBAR
# =====================================================================
st.markdown(f"""
<div class="hero">
  <h1>🫘 Dry Bean Variety Inspector</h1>
  <p>Five machine-learning classifiers that identify a bean's variety from 16 shape measurements.</p>
  <span class="pill">📚 BITS Pilani - M.Tech (AIML)</span>
  <span class="pill">🧠 Machine Learning - Assignment 2</span>
  <span class="pill">👤 {STUDENT_NAME} | {STUDENT_ID}</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("📁 TEST DATA")
uploaded = st.sidebar.file_uploader("Upload a labeled CSV", type=["csv"])
use_bundled = st.sidebar.checkbox("Use bundled test_data.csv", value=True)
if BUNDLED_TEST.exists():
    st.sidebar.download_button("⬇️ Download sample test data", BUNDLED_TEST.read_bytes(),
                               "test_data.csv", "text/csv")
st.sidebar.caption(f"The CSV needs the 16 feature columns and a `{TARGET}` column.")
st.sidebar.markdown("---")
st.sidebar.header("🤖 MODEL")
selected = st.sidebar.selectbox("Select classifier", list(MODEL_FILES))
st.sidebar.info(MODEL_BLURB[selected])
st.sidebar.caption(f"Running scikit-learn {sklearn.__version__}")

# =====================================================================
# LOAD MODELS (needed by every tab)
# =====================================================================
pipes, model_error = {}, None
if MODEL_DIR is None:
    model_error = "No `model/` or `models/` folder was found next to app.py. Commit the saved .joblib files to the repo."
else:
    try:
        pipes = {n: load_pipeline(str(MODEL_DIR / f)) for n, f in MODEL_FILES.items()}
    except Exception as exc:
        model_error = f"Could not load the saved models: {exc}"

FEATURES = list(next(iter(pipes.values())).feature_names_in_) if pipes else list(FEATURE_INFO)
CLASSES = list(next(iter(pipes.values())).classes_) if pipes else list(CLASS_COUNTS)

# =====================================================================
# LOAD DATA
# =====================================================================
data, source = None, "none"
try:
    if uploaded is not None:
        data, source = pd.read_csv(uploaded), uploaded.name
    elif use_bundled and BUNDLED_TEST.exists():
        data, source = pd.read_csv(BUNDLED_TEST), "test_data.csv (bundled)"
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")

tab_about, tab_eval, tab_cmp, tab_data = st.tabs(
    ["🏠 About", "📊 Evaluation", "⚖️ Model comparison", "📂 Data explorer"])

# =====================================================================
# TAB 1 - ABOUT
# =====================================================================
with tab_about:
    st.subheader("🎯 Problem")
    st.markdown("Given 16 size and shape measurements taken from an image of one dry bean, predict which of "
                "**seven varieties** it is. Faster, more consistent sorting is the practical goal.")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("🗂️ Dataset at a glance")
        st.markdown("".join(f'<div class="fact"><span>{k}</span><span><b>{v}</b></span></div>'
                            for k, v in DATASET_FACTS.items()), unsafe_allow_html=True)
    with c2:
        st.subheader("📈 Class balance")
        counts = pd.Series(CLASS_COUNTS).sort_values()
        fig, ax = plt.subplots(figsize=(4.6, 3.1))
        ax.barh(counts.index, counts.values, color="#2f6b3a")
        ax.set_xlabel("Rows in the full dataset")
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("🔠 Feature dictionary")
    st.dataframe(pd.DataFrame({"Feature": FEATURES, "Meaning": [FEATURE_INFO.get(f, "") for f in FEATURES]}),
                 hide_index=True)
    st.subheader("🤖 Models and settings (read from the saved files)")
    if pipes:
        rows = []
        for name, p in pipes.items():
            steps, settings = describe_settings(p)
            rows.append({"Model": name, "Preprocessing steps": steps, "Settings": settings,
                         "How it works": MODEL_BLURB[name]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.warning("Models are not available, so their settings cannot be shown.")
    st.caption("Precision, Recall and F1 use weighted averaging across the 7 classes; AUC is weighted "
               "one-vs-rest. With weighted averaging, Recall equals Accuracy by definition.")

# =====================================================================
# GUARDS FOR THE DATA TABS
# =====================================================================
problem, severity = model_error, st.error
if problem is None and data is None:
    problem, severity = "Upload a labeled CSV or tick **Use bundled test_data.csv** in the sidebar.", st.warning
if problem is None:
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        problem = f"Missing required columns: {missing}"
    elif data.empty:
        problem = "The uploaded file contains no rows."
    else:
        unknown = sorted(set(data[TARGET].astype(str)) - set(CLASSES))
        if unknown:
            problem = (f"The {TARGET} column contains labels the models do not know: {unknown}. "
                       f"Expected values: {CLASSES}")

results = {}
if problem is None:
    try:
        results = {n: score_model(n, data) for n in MODEL_FILES}
    except Exception as exc:
        problem = (f"The models could not score this file ({type(exc).__name__}: {exc}). "
                   "Check that every feature column is numeric.")

if problem:
    for t in (tab_eval, tab_cmp, tab_data):
        with t:
            severity(problem)
else:
    X = data[FEATURES]
    y = data[TARGET].astype(str)

    # =================================================================
    # TAB 2 - EVALUATION
    # =================================================================
    with tab_eval:
        pipe = pipes[selected]
        pred, proba, scores = results[selected]
        correct = int((pred == y.values).sum())

        a, b = st.columns(2)
        a.markdown(f'<div class="card"><h4>🧪 Test data</h4><b>{source}</b><hr>'
                   f'<b>Rows:</b> {len(data):,} &nbsp;|&nbsp; <b>Features:</b> {len(FEATURES)} '
                   f'&nbsp;|&nbsp; <b>Classes in file:</b> {y.nunique()}</div>', unsafe_allow_html=True)
        steps, settings = describe_settings(pipe)
        b.markdown(f'<div class="card"><h4>⚙️ Active model</h4><b>{selected}</b><hr>'
                   f'{settings}</div>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"### {selected} classified **{correct:,}** of **{len(data):,}** beans correctly "
                    f"(**{scores['Accuracy'] * 100:.1f}%**)")
        for col, (k, v) in zip(st.columns(6), scores.items()):
            col.metric(k.upper(), fmt(v))
        st.markdown("---")

        left, right = st.columns(2)
        labels = list(pipe.classes_)
        with left:
            st.subheader("Confusion matrix")
            as_percent = st.checkbox("Show as % of each actual class", value=False)
            cm = confusion_matrix(y, pred, labels=labels)
            if as_percent:
                shown, fmt_str = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100, ".0f"
            else:
                shown, fmt_str = cm, "d"
            fig, ax = plt.subplots(figsize=(6, 4.8))
            sns.heatmap(shown, annot=True, fmt=fmt_str, cmap="YlGn", cbar=False, ax=ax,
                        xticklabels=labels, yticklabels=labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.xticks(rotation=35, ha="right")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.subheader("Classification report")
            rep = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.style.format("{:.3f}"))
            st.markdown("**F1 score per variety**")
            f1s = pd.Series({k: rep[k]["f1-score"] for k in labels}).sort_values()
            fig, ax = plt.subplots(figsize=(5.5, 2.8))
            ax.barh(f1s.index, f1s.values, color="#7b4a12")
            ax.set_xlim(0, 1.05)
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("ROC curves (one variety versus the rest)")
        Y = label_binarize(y, classes=labels)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        drawn = 0
        for i, cls in enumerate(labels):
            if Y.shape[1] > i and 0 < Y[:, i].sum() < len(Y):
                fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
                ax.plot(fpr, tpr, lw=1.6, label=cls)
                drawn += 1
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        if drawn:
            ax.legend(fontsize=8, loc="lower right")
            st.pyplot(fig)
        else:
            st.caption("ROC curves need examples of at least two varieties in the uploaded file.")
        plt.close(fig)

        out = data.copy()
        out["Predicted_Class"] = pred
        out["Confidence"] = proba.max(axis=1).round(3)
        out["Correct"] = pred == y.values
        wrong = out[~out["Correct"]]
        with st.expander(f"Misclassified beans ({len(wrong):,})"):
            st.dataframe(wrong.head(500))
        st.download_button("⬇️ Download all predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                           f"predictions_{selected.lower().replace(' ', '_')}.csv", "text/csv")

    # =================================================================
    # TAB 3 - COMPARISON
    # =================================================================
    with tab_cmp:
        st.subheader("All five models on the selected data")
        comp = pd.DataFrame([{"Model": n, **{k: round(v, 4) for k, v in r[2].items()}}
                             for n, r in results.items()]).set_index("Model")
        best = comp["MCC"].idxmax()
        st.markdown(f'<div class="winner">🏆 <b>Highest MCC: {best}</b> (MCC {fmt(comp.loc[best, "MCC"])}, '
                    f'Accuracy {fmt(comp.loc[best, "Accuracy"])})</div>', unsafe_allow_html=True)
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#2f6b3a66"))
        st.caption("Green cells mark the best value per metric. Models that score within a few samples of each "
                   "other can swap places on a different split, so also look at MCC and per-class results.")
        pick = st.multiselect("Metrics to chart", list(comp.columns), default=["Accuracy", "F1 Score", "MCC"])
        if pick:
            st.bar_chart(comp[pick])

    # =================================================================
    # TAB 4 - DATA EXPLORER
    # =================================================================
    with tab_data:
        st.subheader("Data explorer")
        d1, d2, d3 = st.columns(3)
        d1.metric("Rows", f"{len(data):,}")
        d2.metric("Features", len(FEATURES))
        d3.metric("Classes in file", y.nunique())
        feat = st.selectbox("Distribution of a feature by variety", FEATURES)
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        for cls in sorted(y.unique()):
            ax.hist(X.loc[y == cls, feat], bins=30, alpha=0.5, label=cls)
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(data.head(200))
        with st.expander("Summary statistics"):
            st.dataframe(X.describe().T.round(3))

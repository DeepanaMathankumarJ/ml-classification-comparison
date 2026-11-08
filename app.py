from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import sklearn
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import label_binarize

# =====================================================================
# CONFIG - edit this block only
# =====================================================================

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

STUDENT_NAME = "Deepana Jayachandran"
STUDENT_ID = "2026AK05687"

ROOT = Path(__file__).resolve().parent
MODEL_DIR = next((p for p in (ROOT / "model", ROOT / "models") if p.is_dir()), None)
BUNDLED_TEST = ROOT / "test_data.csv"

# =====================================================================
# PAGE + STYLE
# =====================================================================
st.set_page_config(page_title="Dry Bean Variety Inspector", page_icon="🫘", layout="wide")
# Palette taken from the dashboard reference: sage page, white cards, lime + deep green accents
LIME, DEEP, CORAL = "#9be564", "#05382f", "#ef6a4c"
PALETTE = [DEEP, LIME, "#2f9e8f", CORAL, "#5b7f73", "#c9d96a", "#f2b84b"]
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=PALETTE),
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9aa9a0", "axes.labelcolor": DEEP,
    "xtick.color": DEEP, "ytick.color": DEEP, "text.color": DEEP,
})
GREEN_CMAP = LinearSegmentedColormap.from_list("bean_green", ["#f6fcf0", LIME, DEEP])

st.markdown(f"""
<style>
.stApp {{background: linear-gradient(160deg, #d9efc9 0%, #e3ece3 38%, #e3ece3 70%, #efe6c4 100%);}}
header[data-testid="stHeader"] {{background: transparent;}}
.stApp, .stApp p, .stApp label, .stApp li, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"] {{color: {DEEP};}}
.stApp .hero, .stApp .hero h1, .stApp .hero p {{color: #ffffff;}}
.stApp .hero .pill {{color: {DEEP};}}
label[data-baseweb="checkbox"] span[role="checkbox"][aria-checked="true"], 
div[data-baseweb="checkbox"] > div:first-child {{background-color: {LIME} !important; border-color: {DEEP} !important;}}
input[type="checkbox"]:checked + div {{background-color: {LIME} !important;}}
section[data-testid="stSidebar"] > div {{background: #f4f8f2;}}
.hero {{background: linear-gradient(120deg, {DEEP} 0%, #135b43 100%); padding: 26px 32px;
       border-radius: 24px; color: #ffffff; margin-bottom: 18px;}}
.hero h1 {{margin: 0 0 6px 0; font-size: 2.0rem;}}
.hero p {{margin: 0; font-size: 1.05rem; opacity: 0.93;}}
.pill {{display: inline-block; padding: 4px 14px; margin: 12px 8px 0 0; border-radius: 999px;
       background: {LIME}; font-size: 0.82rem; color: {DEEP}; font-weight: 600;}}
.card {{background: #ffffff; padding: 16px; border-radius: 20px; text-align: center;
       box-shadow: 0 1px 3px rgba(5,56,47,0.06);}}
.card h4 {{margin: 0 0 4px 0;}}
.card hr {{margin: 8px 0; border-color: rgba(5,56,47,0.12);}}
.fact {{display:flex; justify-content:space-between; padding:9px 4px;
       border-bottom:1px dashed rgba(5,56,47,0.25); font-size:0.95rem;}}
.fact span:first-child {{opacity:0.7;}}
.winner {{border-left: 6px solid {LIME}; background: #ffffff;
         padding: 12px 16px; border-radius: 14px; margin: 8px 0 14px 0;}}
.kpi-grid {{display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin: 6px 0 4px 0;}}
@media (max-width: 900px) {{.kpi-grid {{grid-template-columns: repeat(3, minmax(0, 1fr));}}}}
.kpi {{background: #ffffff; border-radius: 18px; padding: 12px 14px; border-top: 5px solid {LIME};
      box-shadow: 0 1px 3px rgba(5,56,47,0.06);}}
.kpi small {{display: block; font-size: 0.72rem; letter-spacing: 0.04em; opacity: 0.7; white-space: nowrap;}}
.kpi b {{font-size: 1.45rem; color: {DEEP};}}
div[data-testid="stMetric"] {{background: #ffffff; border-radius: 18px; padding: 12px 16px;
       box-shadow: 0 1px 3px rgba(5,56,47,0.06);}}
div[data-testid="stVerticalBlockBorderWrapper"] {{background: #ffffff; border: none !important;
       border-radius: 22px; box-shadow: 0 1px 3px rgba(5,56,47,0.06);}}
button[data-baseweb="tab"][aria-selected="true"] {{color: {DEEP}; font-weight: 700;}}
div[data-baseweb="tab-highlight"] {{background-color: {DEEP};}}
.stDownloadButton button {{border-radius: 999px; background: {LIME}; color: {DEEP}; border: none; font-weight: 600;}}
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
    st.markdown("""
**About the dataset**
- **Source:** *Dry Bean Dataset* from the UCI Machine Learning Repository (id 602), published by Koklu and Ozkan (2020).
- **How it was built:** a computer-vision system photographed **13,611 dry beans** with a high-resolution camera; each bean was segmented and measured.
- **Seven varieties (classes):** Barbunya, Bombay, Cali, Dermason, Horoz, Seker and Sira.
- **Sixteen numeric features:** 12 size/dimension measurements (area, perimeter, axis lengths, etc.) and 4 shape-form descriptors (ShapeFactor1 to 4).
- **Class imbalance:** Dermason is the largest class (3,546 beans) and Bombay the smallest (522), so weighted metrics and MCC are reported alongside accuracy.
- **Data quality:** no missing values; 68 duplicate rows were kept. No categorical features, so no encoding is needed.
- **Split:** stratified 80/20 train/test (10,888 / 2,723 rows, seed 42), so every variety appears in both sets in the same proportion.
""")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("🗂️ Dataset at a glance")
        st.markdown("".join(f'<div class="fact"><span>{k}</span><span><b>{v}</b></span></div>'
                            for k, v in DATASET_FACTS.items()), unsafe_allow_html=True)
    with c2:
        st.subheader("📈 Class balance")
        counts = pd.Series(CLASS_COUNTS).sort_values()
        fig, ax = plt.subplots(figsize=(4.6, 3.1), dpi=110)
        ax.barh(counts.index, counts.values, color=DEEP)
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
        tiles = "".join(f'<div class="kpi"><small>{k.upper()}</small><b>{fmt(v)}</b></div>'
                        for k, v in scores.items())
        st.markdown(f'<div class="kpi-grid">{tiles}</div>', unsafe_allow_html=True)
        st.markdown("---")

        labels = list(pipe.classes_)
        rep = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)

        # ---- Row 1: confusion matrix | ROC | F1 per variety (same size, side by side) ----
        as_percent = st.checkbox("Show confusion matrix as % of each actual class", value=False)
        FIG = (4.3, 3.5)
        cm_col, roc_col, f1_col = st.columns(3, gap="medium")

        with cm_col, st.container(border=True):
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y, pred, labels=labels)
            if as_percent:
                shown, fmt_str = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100, ".0f"
            else:
                shown, fmt_str = cm, "d"
            fig, ax = plt.subplots(figsize=FIG, dpi=110)
            sns.heatmap(shown, annot=True, fmt=fmt_str, cmap=GREEN_CMAP, cbar=False, ax=ax,
                        xticklabels=labels, yticklabels=labels, annot_kws={"size": 7})
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with roc_col, st.container(border=True):
            st.subheader("ROC curves")
            Y = label_binarize(y, classes=labels)
            fig, ax = plt.subplots(figsize=FIG, dpi=110)
            drawn = 0
            for i, cls in enumerate(labels):
                if Y.shape[1] > i and 0 < Y[:, i].sum() < len(Y):
                    fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
                    ax.plot(fpr, tpr, lw=1.2, label=cls)
                    drawn += 1
            ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
            ax.set_xlabel("False positive rate", fontsize=8)
            ax.set_ylabel("True positive rate", fontsize=8)
            ax.tick_params(labelsize=7)
            if drawn:
                ax.legend(fontsize=6, loc="lower right")
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.caption("ROC curves need examples of at least two varieties in the uploaded file.")
            plt.close(fig)

        with f1_col, st.container(border=True):
            st.subheader("F1 score per variety")
            f1s = pd.Series({k: rep[k]["f1-score"] for k in labels}).sort_values()
            fig, ax = plt.subplots(figsize=FIG, dpi=110)
            ax.barh(f1s.index, f1s.values, color=[CORAL if i == 0 else LIME for i in range(len(f1s))])
            ax.set_xlim(0, 1.08)
            ax.set_xlabel("F1 score", fontsize=8)
            ax.tick_params(labelsize=7)
            for yi, v in enumerate(f1s.values):
                ax.text(min(v + 0.01, 1.0), yi, f"{v:.2f}", va="center", fontsize=7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # ---- Row 2: full-width report with all six assignment metrics per variety ----
        with st.container(border=True):
            st.subheader("Classification report")
            rows = {}
            for i, cls in enumerate(labels):
                yt, yp = (y.values == cls), (pred == cls)
                try:
                    cls_auc = roc_auc_score(yt, proba[:, i])
                except ValueError:
                    cls_auc = np.nan
                rows[cls] = {
                    "Accuracy": (yt == yp).mean(), "AUC": cls_auc,
                    "Precision": rep[cls]["precision"], "Recall": rep[cls]["recall"],
                    "F1 Score": rep[cls]["f1-score"], "MCC": matthews_corrcoef(yt, yp),
                    "Support": rep[cls]["support"]}
            rows["macro avg"] = {**pd.DataFrame(rows).T.drop(columns="Support").mean().to_dict(),
                                 "Support": len(y)}
            rows["weighted avg (overall)"] = {
                **{k: scores[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]},
                "Support": len(y)}
            report = pd.DataFrame(rows).T
            st.dataframe(
                report.style.format({c: "{:.3f}" for c in report.columns if c != "Support"}
                                    | {"Support": "{:,.0f}"}, na_rep=""),
                height=(len(report) + 1) * 35 + 3)
            st.caption("Per-variety Accuracy, AUC and MCC are one-vs-rest (that variety against all others). "
                       "The last row repeats the overall weighted scores shown in the tiles above.")

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
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color=LIME))
        st.caption("Lime cells mark the best value per metric. Models that score within a few samples of each "
                   "other can swap places on a different split, so also look at MCC and per-class results.")
        pick = st.multiselect("Metrics to chart", list(comp.columns), default=["Accuracy", "F1 Score", "MCC"])
        if pick:
            st.bar_chart(comp[pick], color=PALETTE[:len(pick)])

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
        fig, ax = plt.subplots(figsize=(7.5, 3.0), dpi=110)
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
    
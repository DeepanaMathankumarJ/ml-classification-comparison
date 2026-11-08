# Dry Bean Variety Inspector

**BITS Pilani – M.Tech (AIML) · Machine Learning – Assignment 2**
**Name:** Deepana Jayachandran · **BITS ID:** 2026AK05687

| | |
|---|---|
| **GitHub repository** | `<PASTE YOUR GITHUB REPO URL HERE>` |
| **Live Streamlit app** | `<PASTE YOUR STREAMLIT COMMUNITY CLOUD URL HERE>` |

---

## a. Problem statement

Dry beans of different varieties look very similar to the eye, and sorting them by hand is slow and inconsistent. The goal of this project is a **supervised multi-class classification** model that predicts the **variety of a dry bean (7 classes)** from **16 size and shape measurements** extracted from an image of the bean.

Five classifiers are trained on the same data, evaluated with the same six metrics, and compared. They are then served through an interactive Streamlit web app that loads the saved models (no retraining) and evaluates them on an uploaded test CSV.

## b. Dataset description

| Item | Detail |
|---|---|
| **Name / source** | Dry Bean Dataset, UCI Machine Learning Repository (id 602), Koklu & Ozkan (2020) – https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| **Instances** | 13,611 beans (minimum required: 500) |
| **Features** | 16 numeric (minimum required: 12) – 12 size/dimension measurements and 4 shape-form descriptors |
| **Target** | `Class` – 7 varieties: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| **Missing values** | 0 |
| **Duplicate rows** | 68 (kept) |
| **Class balance** | Imbalanced – DERMASON 3,546 · SIRA 2,636 · SEKER 2,027 · HOROZ 1,928 · CALI 1,630 · BARBUNYA 1,322 · BOMBAY 522 |
| **Train / test split** | Stratified 80 / 20, `random_state=42` → 10,888 training rows / 2,723 test rows |

**Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.

Because the classes are imbalanced, Precision, Recall and F1 are **weighted averages** across the 7 classes, AUC is **weighted one-vs-rest**, and MCC is reported alongside accuracy. (With weighted averaging, Recall is equal to Accuracy by definition.)

## c. GitHub repository

`<PASTE YOUR GITHUB REPO URL HERE>`

```
project-folder/
│-- app.py                      # Streamlit application
│-- requirements.txt            # pinned dependencies
│-- README.md
│-- test_data.csv               # held-out test split (2,723 rows) used for evaluation
│-- DryBean_Model_training.ipynb# data checks, training, evaluation, model export
│-- model/                      # saved scikit-learn pipelines (preprocessing + classifier)
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- gaussian_naive_bayes.joblib
│   └-- svm.joblib
```

## d. Models used

All five models are scikit-learn `Pipeline`s trained on the same stratified training split. Preprocessing lives inside each pipeline, so the app (and any new test CSV) gets exactly the same transformations as at training time.

| Model | Pipeline | Main settings |
|---|---|---|
| Logistic Regression | median imputer → StandardScaler → classifier | `C=10`, `max_iter=3000` |
| Decision Tree | median imputer → classifier | `max_depth=10`, `min_samples_leaf=5` |
| KNN | median imputer → StandardScaler → classifier | `n_neighbors=11`, `weights="distance"` |
| Gaussian Naive Bayes | median imputer → PowerTransformer → classifier | `var_smoothing=1e-9` |
| SVM | median imputer → StandardScaler → classifier | RBF kernel, `C=10`, `gamma="scale"`, `probability=True` |

### Comparison table (test set, 2,723 beans)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9218 | **0.9935** | 0.9228 | 0.9218 | 0.9220 | 0.9055 |
| Decision Tree | 0.9064 | 0.9725 | 0.9063 | 0.9064 | 0.9062 | 0.8867 |
| KNN | 0.9185 | 0.9861 | 0.9192 | 0.9185 | 0.9186 | 0.9014 |
| Naive Bayes (Gaussian) | 0.8972 | 0.9900 | 0.9007 | 0.8972 | 0.8973 | 0.8766 |
| Support Vector Machine | **0.9243** | 0.9933 | **0.9243** | **0.9243** | **0.9243** | **0.9085** |

Precision, Recall and F1 are weighted averages; AUC is weighted one-vs-rest. Best value per column in bold.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best on accuracy, F1 and MCC (0.9218 / 0.9220 / 0.9055) and only 0.0025 behind the SVM, and it has the highest AUC (0.9935). A simple linear model on standardised features separates most varieties well, so the classes are largely linearly separable. It is fast to train, easy to interpret and gives well-ranked probabilities. |
| Decision Tree | Accuracy 0.9064 and the lowest AUC (0.9725) of the five. The depth limit (10) and minimum leaf size (5) keep it from overfitting badly, but each leaf gives only coarse class probabilities, which lowers AUC. Its main strengths are interpretability and no need for feature scaling. |
| KNN | Accuracy 0.9185 and MCC 0.9014, close to Logistic Regression. Distance-weighted voting over 11 neighbours on standardised features works well because beans of one variety cluster together in feature space. Scaling is essential for it, and prediction becomes slower as the training set grows. |
| Naive Bayes | Lowest accuracy, F1 and MCC (0.8972 / 0.8973 / 0.8766). Several features are strongly correlated (for example Area, ConvexArea and EquivDiameter), which breaks the independence assumption. Its AUC (0.9900) is still high, so it ranks the classes well even though its hard predictions are less accurate. The power transform helps the Gaussian assumption but cannot remove the correlation. |
| Support Vector Machine | Best on Accuracy, Precision, Recall, F1 and MCC (0.9243 / 0.9243 / 0.9243 / 0.9243 / 0.9085) with an AUC of 0.9933, within 0.0002 of the best. The RBF kernel captures non-linear class boundaries. Per class, BOMBAY is classified perfectly (F1 1.00) while SIRA is the hardest variety (F1 0.87). It is the slowest of the five to train, and `probability=True` adds extra cost. |
| **Overall winner for this dataset?** | **Support Vector Machine.** It leads on five of the six metrics, including weighted F1 (the main selection criterion) and MCC, which is the most informative single score on imbalanced multi-class data. Logistic Regression is a very close second and has the marginally higher AUC (0.9935 vs 0.9933), so it is a good lightweight alternative. The gap between the best and worst model is only about 2.7 accuracy points, so all five are usable on this dataset. |

## Streamlit application

The app is wide-layout with four tabs:

1. **About** – problem statement, dataset facts and class balance, feature dictionary, and the settings of each saved model (read from the `.joblib` files).
2. **Evaluation** – for the selected model: all six metrics, confusion matrix (counts or % per actual class), one-vs-rest ROC curves, F1 per variety, and a full classification report with Accuracy, AUC, Precision, Recall, F1, MCC and Support per variety. Misclassified beans can be inspected and all predictions downloaded as CSV.
3. **Model comparison** – the six metrics for all five models side by side, with a chart.
4. **Data explorer** – row count, feature distributions by variety, a data preview and summary statistics.

**Required features implemented**

- Dataset upload (CSV) – upload a labelled test CSV, or tick *Use bundled test_data.csv* in the sidebar.
- Model selection dropdown – choose any of the five classifiers.
- Evaluation metrics – Accuracy, AUC, Precision, Recall, F1 and MCC.
- Confusion matrix and classification report.
- No retraining – the app only loads the saved pipelines and predicts.

The uploaded CSV must contain the 16 feature columns and the `Class` column.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

`scikit-learn` is pinned in `requirements.txt` to the version used for training, because saved `.joblib` pipelines are version-sensitive. To regenerate the models and `test_data.csv`, run `DryBean_Model_training.ipynb` from top to bottom.

## Academic integrity

The code, UI, analysis and observations in this repository are my own work for this assignment. Any AI tooling was used for learning support and review, not for direct copy-paste submission.

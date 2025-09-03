# disease_prediction_pipeline_fixed.py
"""
Disease prediction pipeline supporting:
 - Logistic Regression
 - SVM
 - Random Forest
 - XGBoost (optional)
Compatible with latest scikit-learn and Python 3.13
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Optional packages
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except:
    IMBLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# ----------------------------
# Settings
# ----------------------------
DATASET = "breast_cancer"   # options: "breast_cancer", "csv"
DATA_PATH = "pima_diabetes.csv"  # used if DATASET == "csv"
TARGET_COL = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.25
USE_SMOTE = False

# ----------------------------
# Load dataset
# ----------------------------
def load_dataset(dataset_name):
    if dataset_name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        df.columns = list(data.feature_names) + ['target']
        df['target'] = data.target
        print("Loaded sklearn breast_cancer dataset. Shape:", df.shape)
        return df
    elif dataset_name == "csv":
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        print("Loaded CSV file. Shape:", df.shape)
        return df
    else:
        raise ValueError("Unsupported DATASET. Choose 'breast_cancer' or 'csv'")

# ----------------------------
# Basic feature engineering
# ----------------------------
def basic_feature_engineering(df):
    df = df.copy()
    # Example: BMI per age or glucose/insulin ratio (only if columns exist)
    if {"age", "bmi"}.issubset(df.columns):
        df["bmi_per_age"] = df["bmi"] / (df["age"].replace(0, np.nan))
        df["bmi_per_age"].fillna(df["bmi_per_age"].median(), inplace=True)
    if {"glucose", "insulin"}.issubset(df.columns):
        df["insulin_glucose_ratio"] = df["insulin"] / (df["glucose"].replace(0, np.nan))
        df["insulin_glucose_ratio"].fillna(df["insulin_glucose_ratio"].median(), inplace=True)
    return df

# ----------------------------
# Utility to get numeric/categorical features
# ----------------------------
def get_feature_types(df, target_col):
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical:
        numerical.remove(target_col)
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numerical, categorical

# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline():
    df = load_dataset(DATASET)
    target = TARGET_COL if DATASET == "csv" else "target"

    # Basic FE
    df = basic_feature_engineering(df)

    df = df[~df[target].isna()].copy()
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    numeric_features, categorical_features = get_feature_types(X, target)

    # --------------------
    # Fix column names for sklearn
    # --------------------
    X = X.rename(columns=lambda x: str(x).strip().replace(' ', '_'))
    numeric_features = [col.strip().replace(' ', '_') for col in numeric_features]
    categorical_features = [col.strip().replace(' ', '_') for col in categorical_features]

    # Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # fixed!
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')

    X_train_prep = preprocessor.fit_transform(X_train_raw)
    X_test_prep = preprocessor.transform(X_test_raw)

    # Optional SMOTE
    if USE_SMOTE and IMBLEARN_AVAILABLE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_prep, y_train = sm.fit_resample(X_train_prep, y_train)

    # Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE)
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train_prep, y_train)
        y_pred = model.predict(X_test_prep)
        y_proba = model.predict_proba(X_test_prep)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test_prep)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)

        results[name] = {"model": model, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

        print(f"{name} metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc:.4f}")
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save best model by ROC-AUC
    best_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_name]['model']
    final_pipeline = Pipeline([('preprocessor', preprocessor), ('model', best_model)])
    joblib.dump(final_pipeline, f"best_disease_model_{best_name}.joblib")
    print(f"\nSaved best pipeline as: best_disease_model_{best_name}.joblib")

    return results, final_pipeline

# Run
if __name__ == "__main__":
    results, pipeline = run_pipeline()

 import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Optional: XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Load dataset
def load_heart_disease_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv(url, header=None, names=columns)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# Preprocessing
def preprocess_data(df):
    X = df.drop(columns=['target'])
    y = df['target'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification: 1 = disease, 0 = no disease
    X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Train models
def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100),
    }
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(eval_metric='logloss')
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

# Evaluate models
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba)
        }
    return pd.DataFrame(results).T

# Plot bar charts
def plot_metrics(metrics_df):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metrics_df.index, y=metrics_df[metric])
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Main pipeline
def main():
    df = load_heart_disease_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    models = train_models(X_train, y_train)
    metrics_df = evaluate_models(models, X_test, y_test)
    print(metrics_df)
    plot_metrics(metrics_df)

    # Save the best model
    best_model_name = metrics_df['ROC AUC'].idxmax()
    best_model = models[best_model_name]
    joblib.dump(best_model, f'best_model_{best_model_name}.joblib')
    print(f"Best model saved as 'best_model_{best_model_name}.joblib'")

if __name__ == "__main__":
    main()


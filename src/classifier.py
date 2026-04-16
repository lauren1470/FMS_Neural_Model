"""
ML Classification Pipeline for FMS Neural Model

Trains Random Forest and SVM classifiers to distinguish healthy vs
fibromyalgia firing patterns from the synthetic spike-train dataset.

Usage:
    python src/classifier.py                         # default: data/dataset.csv
    python src/classifier.py --data path/to/data.csv # custom dataset path
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from joblib import dump
from matplotlib.colors import LinearSegmentedColormap


# --- Project colour palette ---
FMS_CMAP = LinearSegmentedColormap.from_list(
    'fms', ['#A8C4E0', '#4472C4', '#9B59B6', '#E91E63']
)

# --- Columns to drop before training (metadata, not features) ---
DROP_COLS = ['label', 'seed', 'protocol']


def load_and_split(csv_path: str, test_size: float = 0.3, seed: int = 42):
    """Load dataset, separate features/labels, and perform stratified split."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"  Classes: {df['label'].value_counts().to_dict()}")

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)} (stratified {int((1-test_size)*100)}/{int(test_size*100)} split)")

    return X_train, X_test, y_train, y_test, feature_cols


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """Fit StandardScaler on training data and transform both sets."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def train_and_evaluate(name: str, model, X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray):
    """Train a classifier, print results, and return predictions + metrics."""
    
    print(f"  {name}")
    

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='fibromyalgia')
    rec = recall_score(y_test, y_pred, pos_label='fibromyalgia')
    f1 = f1_score(y_test, y_pred, pos_label='fibromyalgia')

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    return y_pred, metrics


def cross_validate_model(name: str, model, X: np.ndarray, y: np.ndarray,
                         n_folds: int = 5, seed: int = 42):
    """Run stratified k-fold cross-validation and report results."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)

    print(f"\n  {name} — {n_folds}-fold Cross-Validation:")
    print(f"    Accuracy:  {results['test_accuracy'].mean():.4f} (+/- {results['test_accuracy'].std():.4f})")
    print(f"    Precision: {results['test_precision_macro'].mean():.4f} (+/- {results['test_precision_macro'].std():.4f})")
    print(f"    Recall:    {results['test_recall_macro'].mean():.4f} (+/- {results['test_recall_macro'].std():.4f})")
    print(f"    F1-score:  {results['test_f1_macro'].mean():.4f} (+/- {results['test_f1_macro'].std():.4f})")

    return {k.replace('test_', ''): float(v.mean()) for k, v in results.items() if k.startswith('test_')}


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    """Run GridSearchCV to find optimal hyperparameters for RF and SVM."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # --- Random Forest ---
    print("\n  Tuning Random Forest...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=seed, n_jobs=-1),
        rf_params, cv=cv, scoring='f1_macro', n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    print(f"    Best params: {rf_grid.best_params_}")
    print(f"    Best CV F1:  {rf_grid.best_score_:.4f}")

    # --- SVM ---
    print("\n  Tuning SVM (RBF)...")
    svm_params = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.01, 0.1],
    }
    svm_grid = GridSearchCV(
        SVC(kernel='rbf', random_state=seed),
        svm_params, cv=cv, scoring='f1_macro', n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)
    print(f"    Best params: {svm_grid.best_params_}")
    print(f"    Best CV F1:  {svm_grid.best_score_:.4f}")

    return rf_grid.best_estimator_, svm_grid.best_estimator_, rf_grid.best_params_, svm_grid.best_params_


def plot_confusion_matrix(y_test, y_pred, name: str, output_dir: str):
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=['healthy', 'fibromyalgia'])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=FMS_CMAP,
                xticklabels=['Healthy', 'FMS'],
                yticklabels=['Healthy', 'FMS'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{name} — Confusion Matrix')
    fig.tight_layout()
    path = os.path.join(output_dir, f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(importances: np.ndarray, feature_names: list, output_dir: str):
    """Save a horizontal bar chart of RF feature importances."""
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    n_bars = len(idx)
    bar_colours = [FMS_CMAP(i / max(n_bars - 1, 1)) for i in range(n_bars)]
    ax.barh(range(n_bars), importances[idx], color=bar_colours)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Importance (Gini)')
    ax.set_title('Random Forest — Feature Importance')
    fig.tight_layout()
    path = os.path.join(output_dir, 'feature_importance_rf.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_model_comparison(rf_metrics: dict, svm_metrics: dict, output_dir: str):
    """Save a grouped bar chart comparing RF vs SVM metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    rf_vals = [rf_metrics[m] for m in metrics]
    svm_vals = [svm_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='#4472C4')
    ax.bar(x + width/2, svm_vals, width, label='SVM (RBF)', color='#E91E63')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Test Set Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'model_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='FMS ML Classifier')
    parser.add_argument('--data', default='data/dataset.csv', help='Path to dataset CSV')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    args = parser.parse_args()

    tag = 'tuned' if args.tune else 'initial'
    figures_dir = os.path.join('figures', tag)
    models_dir = os.path.join('models', tag)
    results_dir = os.path.join('results', tag)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- 1. Load & Split ---
    X_train, X_test, y_train, y_test, feature_names = load_and_split(
        args.data, test_size=0.3, seed=args.seed
    )

    # --- 2. Scale ---
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # --- 3. Define or tune models ---
    if args.tune:
        print("\n  Hyperparameter Tuning")
        rf, svm, rf_best_params, svm_best_params = tune_hyperparameters(
            X_train_s, y_train, seed=args.seed
        )
    else:
        rf_best_params = None
        svm_best_params = None
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            random_state=args.seed,
            n_jobs=-1
        )
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=args.seed
        )

    # --- 4. Train & Evaluate on test set ---
    rf_pred, rf_metrics = train_and_evaluate(
        'Random Forest', rf, X_train_s, X_test_s, y_train, y_test
    )
    svm_pred, svm_metrics = train_and_evaluate(
        'SVM (RBF Kernel)', svm, X_train_s, X_test_s, y_train, y_test
    )

    # --- 5. Cross-validation (training data only to avoid data leakage) ---
    print(f"  Cross-Validation")

    rf_cv = cross_validate_model('Random Forest', RandomForestClassifier(
        n_estimators=200, min_samples_split=5, random_state=args.seed, n_jobs=-1
    ), X_train_s, y_train)

    svm_cv = cross_validate_model('SVM (RBF)', SVC(
        kernel='rbf', C=1.0, gamma='scale', random_state=args.seed
    ), X_train_s, y_train)

    # --- 6. Plots ---
    print(f"  Saving Figures")

    plot_confusion_matrix(y_test, rf_pred, 'Random Forest', figures_dir)
    plot_confusion_matrix(y_test, svm_pred, 'SVM (RBF)', figures_dir)
    plot_feature_importance(rf.feature_importances_, feature_names, figures_dir)
    plot_model_comparison(rf_metrics, svm_metrics, figures_dir)

    # --- 7. Save models & results ---
    dump(rf, os.path.join(models_dir, 'random_forest.joblib'))
    dump(svm, os.path.join(models_dir, 'svm_rbf.joblib'))
    dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    print(f"\n  Models saved to {models_dir}/")

    results = {
        'random_forest': {
            'test_set': rf_metrics,
            'cross_validation': rf_cv,
            'feature_importance': dict(zip(feature_names, rf.feature_importances_.tolist())),
            'best_params': rf_best_params
        },
        'svm_rbf': {
            'test_set': svm_metrics,
            'cross_validation': svm_cv,
            'best_params': svm_best_params
        }
    }
    results_path = os.path.join(results_dir, 'classification_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")


if __name__ == '__main__':
    main()

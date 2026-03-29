#!/usr/bin/env python3
"""
Trains 4 classifiers on the original (unaugmented) images.
SVM, SGD are the baselines from phase 1. Random Forest and
XGBoost are the two additional ones I added.

Does StandardScaler + PCA first, then trains and pickles everything.
"""

import os
import sys
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_dataset


def train_models(X_train, y_train):
    """Train all 4 and return them as a dict."""
    models = {}

    print("\n--- Training SVM ---")
    t = time.time()
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    print(f"    Done in {time.time()-t:.1f}s")
    models['SVM'] = svm

    print("\n--- Training SGD ---")
    t = time.time()
    sgd = SGDClassifier(loss='log_loss', alpha=0.01, max_iter=1000, random_state=42)
    sgd.fit(X_train, y_train)
    print(f"    Done in {time.time()-t:.1f}s")
    models['SGD'] = sgd

    print("\n--- Training Random Forest ---")
    t = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"    Done in {time.time()-t:.1f}s")
    models['Random Forest'] = rf

    print("\n--- Training XGBoost ---")
    t = time.time()
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train)
    print(f"    Done in {time.time()-t:.1f}s")
    models['XGBoost'] = xgb_clf

    return models


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'original_pdfs')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Classifier Training")
    print("=" * 60)
    print(f"Data: {data_dir}")

    X, y = load_dataset(data_dir)

    # normalize
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # reduce dimensionality - 40k features is way too many
    print("Applying PCA (40,000 -> 100 components)...")
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var = sum(pca.explained_variance_ratio_) * 100
    print(f"  Variance explained: {var:.1f}%")

    # 80/20 split
    print("\nSplitting 80/20 (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    models = train_models(X_train, y_train)

    # save everything
    print("\nSaving models...")
    with open(os.path.join(results_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(results_dir, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    for name, model in models.items():
        fname = name.replace(' ', '_').lower() + '_model.pkl'
        with open(os.path.join(results_dir, fname), 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved: {fname}")

    print("\nDone! Run analysis.py next.")


if __name__ == "__main__":
    main()

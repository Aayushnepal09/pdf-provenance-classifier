#!/usr/bin/env python3
"""
Runs the full analysis - evaluates all 4 classifiers on original
images plus each augmentation type separately. Makes confusion
matrices, robustness plot, and does McNemar's significance test.
"""

import os
import sys
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_dataset, load_augmented_dataset


def load_models_and_transforms(models_dir):
    """Load the saved models and preprocessing stuff from results/"""
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, 'pca.pkl'), 'rb') as f:
        pca = pickle.load(f)

    models = {}
    for name in ['svm', 'sgd', 'random_forest', 'xgboost']:
        path = os.path.join(models_dir, f"{name}_model.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        else:
            print(f"Warning: {name}_model.pkl not found")

    return models, scaler, pca


def plot_confusion_matrix(y_true, y_pred, title, output_path):
    """Makes a confusion matrix heatmap and saves it."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Word', 'Google', 'Python'],
        yticklabels=['Word', 'Google', 'Python']
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def plot_robustness_curves(df, output_path):
    """Makes a line chart comparing accuracy across conditions for each model."""
    plt.figure(figsize=(10, 6))
    for model in df['Model'].unique():
        subset = df[df['Model'] == model]
        plt.plot(subset['Condition'], subset['Accuracy'],
                 marker='o', linewidth=2, label=model)

    plt.title('Classifier Robustness Across Augmentations')
    plt.xlabel('Condition')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test - checks if two classifiers disagree significantly.
    Returns (test_statistic, p_value).
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # b = A got right but B got wrong, c = opposite
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return 0.0, 1.0

    # with continuity correction
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return round(stat, 4), round(p, 4)


def run_significance_tests(predictions, y_true, output_path):
    """Pairwise McNemar tests between all classifier pairs."""
    names = list(predictions.keys())
    pairs = list(itertools.combinations(names, 2))

    rows = []
    print("\n--- McNemar's Significance Tests (original data) ---")
    print(f"{'Model A':<20} {'Model B':<20} {'Stat':>8} {'p-val':>8} {'Sig?':>10}")
    print("-" * 70)

    for a, b in pairs:
        stat, p = mcnemar_test(y_true, predictions[a], predictions[b])
        sig = "YES" if p < 0.05 else "no"
        print(f"{a:<20} {b:<20} {stat:>8} {p:>8} {sig:>10}")
        rows.append({
            'Model A': a, 'Model B': b,
            'McNemar Statistic': stat,
            'p-value': p,
            'Significant (p<0.05)': p < 0.05
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_dir = os.path.join(base_dir, 'data', 'original_pdfs')
    aug_dir = os.path.join(base_dir, 'data', 'augmented_images')
    results_dir = os.path.join(base_dir, 'results')

    cm_dir = os.path.join(results_dir, 'confusion_matrices')
    plots_dir = os.path.join(results_dir, 'robustness_plots')
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 60)
    print("Analysis & Robustness Testing")
    print("=" * 60)

    models, scaler, pca = load_models_and_transforms(results_dir)
    print(f"Loaded models: {list(models.keys())}")

    conditions = ['original', 'gaussian', 'jpeg', 'dpi', 'crop', 'bitdepth']
    all_results = []
    orig_preds = {}  # need to save these for mcnemar later
    orig_y = None

    for cond in conditions:
        print(f"\n--- {cond.upper()} ---")

        if cond == 'original':
            X, y = load_dataset(orig_dir)
        else:
            X, y = load_augmented_dataset(aug_dir, cond)

        if len(X) == 0:
            print(f"  No data for '{cond}', skipping.")
            continue

        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        for model_name, model in models.items():
            y_pred = model.predict(X_pca)

            acc = accuracy_score(y, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y, y_pred, average='macro', zero_division=0
            )

            print(f"  {model_name:<15} acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

            all_results.append({
                'Model': model_name,
                'Condition': cond,
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1 Score': round(f1, 4)
            })

            # confusion matrix
            cm_file = os.path.join(cm_dir, f"cm_{model_name.replace(' ', '_')}_{cond}.png")
            plot_confusion_matrix(y, y_pred, f"{model_name} - {cond}", cm_file)

            # save predictions for significance test (original only)
            if cond == 'original':
                orig_preds[model_name] = y_pred
                orig_y = y

    # save metrics
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(results_dir, 'performance_metrics.csv'), index=False)
    print(f"\nMetrics saved to results/performance_metrics.csv")

    # robustness plot
    plot_robustness_curves(df, os.path.join(plots_dir, 'robustness_curves.png'))
    print("Robustness curves saved.")

    # significance tests
    if orig_y is not None and len(orig_preds) > 1:
        run_significance_tests(orig_preds, orig_y,
                              os.path.join(results_dir, 'significance_tests.csv'))

    print("\nDone! Check the results/ folder.")


if __name__ == "__main__":
    main()

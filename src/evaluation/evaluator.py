"""
Model Evaluation and Diagnostic Reporting.
Computes accuracy, confusion matrix decomposition, full classification report,
and visual benchmark charts replicating IAS_Classificacao.ipynb.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.tree import plot_tree


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes full set of classification metrics for true vs. predicted targets."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, zero_division=0)
    }


def explain_confusion_matrix(cm: np.ndarray) -> Dict[str, int]:
    """
    Deconstructs a binary confusion matrix into its 4 canonical components:
    - TN (True Negative): Actual 0 predicted as 0
    - FP (False Positive / Type I Error): Actual 0 predicted as 1
    - FN (False Negative / Type II Error): Actual 1 predicted as 0
    - TP (True Positive): Actual 1 predicted as 1
    """
    if cm.shape != (2, 2):
        raise ValueError("explain_confusion_matrix requires a 2x2 binary matrix.")
    tn, fp, fn, tp = cm.ravel()
    return {
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)
    }


def generate_comparison_table(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Builds a summary comparison DataFrame across all models, matching
    the summary table in IAS_Classificacao.ipynb.
    """
    records = []
    for model_name, y_pred in predictions_dict.items():
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        records.append({
            'Modelo': model_name,
            'Acurácia': round(acc, 4),
            'Precisão': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4)
        })
    return pd.DataFrame(records).sort_values(by='Acurácia', ascending=False).reset_index(drop=True)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'Acurácia',
    title: str = 'Comparação entre Modelos',
    save_path: Optional[str] = None
) -> None:
    """Generates a bar chart comparing models on a given metric."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(comparison_df['Modelo'], comparison_df[metric], color=['#2b5c8f', '#4682b4', '#5f9ea0', '#87ceeb'])
    plt.ylabel(metric)
    plt.ylim(0, 1.1)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')
                     
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_decision_tree_structure(
    tree_model: Any,
    feature_names: List[str],
    class_names: List[str] = ['Não', 'Sim'],
    save_path: Optional[str] = None
) -> None:
    """Visualizes the hierarchical decision rules of a trained Decision Tree."""
    plt.figure(figsize=(18, 10))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title('Árvore de Classificação - Regras Hierárquicas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
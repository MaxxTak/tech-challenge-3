"""
Visualization and Diagnostic Charts Module.
Centralizes generation of model comparisons, confusion matrix heatmaps,
feature importance rankings, SHAP summaries, and tree structures.
"""

from typing import List, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import shap


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'F1-Score',
    title: str = 'Comparação de Modelos',
    save_path: Optional[str] = None
) -> None:
    """Generates a styled bar chart comparing models on a specific metric."""
    plt.figure(figsize=(9, 5))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    
    # Handle possible string formatting in metric
    metric_series = pd.to_numeric(comparison_df[metric], errors='coerce')

    bars = plt.bar(
        comparison_df['Modelo'],
        metric_series,
        color=colors[:len(comparison_df)]
    )
    plt.ylabel(metric, fontsize=12, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.annotate(
                f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=10
            )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str] = ['Abaixo Meta', 'Atingiu Meta'],
    title: str = 'Matriz de Confusão',
    save_path: Optional[str] = None
) -> None:
    """Renders a confusion matrix heatmap with counts and percentages."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    total = np.sum(cm)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total) * 100
            plt.text(
                j, i, f"{val}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontweight='bold', fontsize=11
            )

    plt.ylabel('Real', fontsize=11, fontweight='bold')
    plt.xlabel('Predito', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importance(
    df_importance: pd.DataFrame,
    top_n: int = 10,
    title: str = 'Top Features - Importância Global',
    save_path: Optional[str] = None
) -> None:
    """Plots horizontal bar chart of the top N most important features."""
    top_df = df_importance.head(top_n).sort_values(by='Valor', ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_df['Feature'], top_df['Valor'], color='#2b5c8f')
    plt.xlabel(top_df['Métrica'].iloc[0] if 'Métrica' in top_df else 'Score', fontsize=11, fontweight='bold')
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        plt.annotate(
            f'{width:.3f}',
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha='left', va='center', fontsize=9, fontweight='bold'
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_shap_summary(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: List[str],
    max_display: int = 10,
    save_path: Optional[str] = None
) -> None:
    """Generates and saves a SHAP summary beeswarm/bar plot."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.title('Explicabilidade Global do Modelo (SHAP Values)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_decision_tree_structure(
    tree_model: Any,
    feature_names: List[str],
    class_names: List[str] = ['Não Atingiu', 'Atingiu'],
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
    plt.title('Árvore de Decisão - Regras Hierárquicas de Classificação', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_regression_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Modelo",
    save_path: Optional[str] = None
) -> None:
    """
    Diagnostic for regression:
    1. Real vs Predicted scatter plot (with ideal y=x line)
    2. Residuals vs Predicted scatter plot
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Real vs Predito
    axes[0].scatter(y_true, y_pred, alpha=0.7, color='teal', edgecolors='k')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Linha Ideal (y = x)')
    axes[0].set_xlabel('Valor Real', fontsize=11)
    axes[0].set_ylabel('Valor Predito', fontsize=11)
    axes[0].set_title(f'{model_name}: Real vs Predito', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Residuos vs Predito
    axes[1].scatter(y_pred, residuals, alpha=0.7, color='crimson', edgecolors='k')
    axes[1].axhline(0, color='black', linestyle='--', lw=2)
    axes[1].set_xlabel('Valor Predito', fontsize=11)
    axes[1].set_ylabel('Resíduo (Real - Predito)', fontsize=11)
    axes[1].set_title(f'{model_name}: Análise de Resíduos', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_data_scatter(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str = "Feature",
    y_label: str = "Target",
    title: str = "Dispersão dos Dados",
    save_path: Optional[str] = None
) -> None:
    """Plots a simple scatter plot of one feature vs the target to visualize the data distribution."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, color='#2ca02c', edgecolors='k')
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

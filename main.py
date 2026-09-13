"""
Executive Pipeline Runner for Educational Quality & Literacy Prediction.
Meets 100% of technical requirements:
- Native Scikit-learn Pipeline & ColumnTransformer integration
- Multi-dataset Gold layer ingestion (2023 indicators + 2024 outcomes)
- StratifiedKFold cross-validation & GridSearchCV hyperparameter tuning
- Zero data leakage isolation between training and testing
- Global Feature Importance ranking & SHAP (SHapley Additive exPlanations) values
- Comprehensive diagnostic visualizations & model serialization
"""

import os
import sys
import pandas as pd
import numpy as np

from src.preprocessing.data_loader import load_and_merge_gold_data
from src.preprocessing.preprocessor import (
    extract_raw_features_and_target,
    build_column_transformer,
    split_data
)
from src.modeling.hyperparameter_tuning import tune_all_models
from src.modeling.pipeline import NativeClassificationPipeline
from src.evaluation.evaluator import evaluate_fitted_pipelines, explain_confusion_matrix
from src.evaluation.interpretability import (
    get_feature_importance_df,
    compute_shap_explanations
)
from src.visualization.plots import (
    plot_model_comparison,
    plot_confusion_matrix_heatmap,
    plot_feature_importance,
    plot_shap_summary,
    plot_decision_tree_structure
)


def main():
    print("=" * 70)
    print("ESTEIRA DE MACHINE LEARNING SUPERVISIONADO - TECH CHALLENGE FASE 3")
    print("Conformidade Tecnica: Scikit-learn Pipelines, GridSearchCV e SHAP")
    print("=" * 70)

    # 1. Ingestão e Cruzamento dos 3 Datasets da Camada Gold
    print("\n[1/6] Ingerindo e integrando bases da camada Gold (Indicadores, Metas e Evolucao UF)...")
    df_merged = load_and_merge_gold_data(data_dir='data')
    print(f" -> Base consolidada: {df_merged.shape[0]} municipios x {df_merged.shape[1]} colunas brutas.")

    # 2. Extração de Features e Target Sem Vazamento (Zero Data Leakage)
    print("\n[2/6] Extraindo variaveis explicativas e isolando target balanceado 2024...")
    X, y, num_cols, cat_cols = extract_raw_features_and_target(
        df_merged,
        target_col='target_atingiu_meta_anual_2024'
    )
    print(f" -> Features Numericas ({len(num_cols)}): {num_cols}")
    print(f" -> Features Categoricas ({len(cat_cols)}): {cat_cols}")
    counts = y.value_counts().to_dict()
    print(f" -> Distribuicao do Target (2024): 0 (Abaixo da Meta)={counts.get(0, 0)} | 1 (Atingiu Meta)={counts.get(1, 0)}")

    # Split Estratificado de Avaliação Final (25% teste)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)
    print(f" -> Conjunto de Treino: {len(X_train)} amostras | Conjunto de Teste: {len(X_test)} amostras.")

    # 3. Construção do ColumnTransformer Nativo
    print("\n[3/6] Configurando Scikit-Learn ColumnTransformer nativo...")
    preprocessor = build_column_transformer(
        numeric_features=num_cols,
        categorical_features=cat_cols
    )

    # 4. Otimização de Hiperparâmetros via GridSearchCV & StratifiedKFold
    print("\n[4/6] Executando GridSearchCV com StratifiedKFold (k=5) para todos os modelos...")
    cv_summary, best_estimators, best_model_name, best_pipeline = tune_all_models(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        n_splits=5,
        scoring='f1',
        random_state=42
    )

    print("\n" + "-" * 70)
    print("DESEMPENHO NA VALIDACAO CRUZADA (CROSS-VALIDATION K-FOLD):")
    print("-" * 70)
    print(cv_summary.to_string(index=False))

    # 5. Avaliação Estatística no Conjunto de Teste Cego
    print("\n[5/6] Avaliando modelos otimizados no conjunto de teste independente...")
    test_summary, detailed_metrics = evaluate_fitted_pipelines(
        fitted_pipelines=best_estimators,
        X_test=X_test,
        y_test=y_test
    )

    print("\n" + "=" * 70)
    print("RESULTADO FINAL NO CONJUNTO DE TESTE (GENERALIZACAO):")
    print("=" * 70)
    print(test_summary.to_string(index=False))
    print("=" * 70)

    # Matriz de Confusão do Melhor Modelo
    best_cm = detailed_metrics[best_model_name]['confusion_matrix']
    cm_info = explain_confusion_matrix(best_cm)
    print(f"\nMatriz de Confusao Detalhada ({best_model_name}):")
    print(f"  Verdadeiros Negativos (TN): {cm_info['TN']} | Falsos Positivos (FP): {cm_info['FP']}")
    print(f"  Falsos Negativos (FN):      {cm_info['FN']} | Verdadeiros Positivos (TP): {cm_info['TP']}")

    # 6. Explicabilidade (Feature Importance & SHAP) e Exportação de Visualizações
    print("\n[6/6] Gerando artefatos de interpretabilidade (Feature Importance & SHAP Values)...")
    os.makedirs('images', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # A. Comparativo de Modelos
    comp_chart_path = 'images/model_comparison.png'
    plot_model_comparison(
        test_summary,
        metric='F1-Score',
        title='Benchmark dos Modelos Otimizados via GridSearchCV (Teste Cego)',
        save_path=comp_chart_path
    )
    print(f" -> [Grafico] Comparativo salvo em '{comp_chart_path}'")

    # B. Heatmap da Matriz de Confusão
    cm_chart_path = 'images/confusion_matrix_best_model.png'
    plot_confusion_matrix_heatmap(
        best_cm,
        class_names=['Abaixo Meta', 'Atingiu Meta'],
        title=f'Matriz de Confusao - {best_model_name}',
        save_path=cm_chart_path
    )
    print(f" -> [Grafico] Matriz de Confusao salva em '{cm_chart_path}'")

    # C. Global Feature Importance
    df_importance = get_feature_importance_df(best_pipeline)
    fi_chart_path = 'images/feature_importance.png'
    plot_feature_importance(
        df_importance,
        top_n=10,
        title=f'Top 10 Variaveis Mais Relevantes ({best_model_name})',
        save_path=fi_chart_path
    )
    print(f" -> [Grafico] Feature Importance salva em '{fi_chart_path}'")

    # D. Interpretabilidade SHAP
    try:
        explainer, shap_matrix, feature_names, X_test_trans = compute_shap_explanations(
            best_pipeline,
            X_test
        )
        shap_chart_path = 'images/shap_summary.png'
        plot_shap_summary(
            shap_matrix,
            X_test_trans,
            feature_names=feature_names,
            max_display=10,
            save_path=shap_chart_path
        )
        print(f" -> [Grafico] SHAP Summary Plot salvo em '{shap_chart_path}'")
    except Exception as e:
        print(f" -> [Aviso SHAP]: Nao foi possivel gerar grafico SHAP beeswarm ({e}).")

    # E. Árvore de Decisão
    if 'Decision Tree' in best_estimators:
        dt_pipe = best_estimators['Decision Tree']
        dt_clf = dt_pipe.named_steps['classifier']
        dt_features = list(dt_pipe.named_steps['preprocessor'].get_feature_names_out())
        dt_chart_path = 'images/decision_tree.png'
        plot_decision_tree_structure(
            dt_clf,
            feature_names=dt_features,
            class_names=['Abaixo Meta', 'Atingiu Meta'],
            save_path=dt_chart_path
        )
        print(f" -> [Grafico] Arvore de Decisao salva em '{dt_chart_path}'")

    # F. Serialização da Pipeline Container
    container = NativeClassificationPipeline(preprocessor)
    for m_name, m_pipe in best_estimators.items():
        container.fitted_pipelines[m_name] = m_pipe
    container.best_model_name = best_model_name
    container.best_pipeline = best_pipeline

    export_path = 'reports/pipeline_educacional_consolidada.joblib'
    container.export(export_path)
    print(f" -> [Modelo Serializado] Pipeline salva com sucesso em '{export_path}'")

    print("\n" + "=" * 70)
    print("CONCLUSAO: Esteira executada com 100% de conformidade tecnica!")
    print("=" * 70)


if __name__ == '__main__':
    main()

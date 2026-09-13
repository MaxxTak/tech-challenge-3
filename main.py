"""
Main Pipeline Runner for Educational Classification Models.
Unifies all 3 Gold layer datasets (indicators, goals, and state evolution),
prepares balanced targets, trains the 4 foundational classifiers, and exports visual diagnostics.
"""

import os
import sys
import pandas as pd
from src.preprocessing.data_loader import load_and_merge_gold_data
from src.preprocessing.preprocessor import prepare_features_and_target_for_modeling
from src.modeling.pipeline import ClassificationPipeline
from src.evaluation.evaluator import plot_model_comparison, plot_decision_tree_structure


def main():
    print("=" * 60)
    print("INICIANDO PIPELINE DE CLASSIFICACAO EDUCACIONAL INTEGRADA")
    print("=" * 60)

    # 1. Carregamento e Cruzamento dos 3 Datasets Gold
    print("\n[1/5] Carregando e unificando os 3 datasets da camada Gold...")
    df_merged = load_and_merge_gold_data(data_dir='data')
    print(f" -> Base consolidada criada com {df_merged.shape[0]} registros e {df_merged.shape[1]} colunas.")

    # 2. Verificação e Distribuição do Target Balanceado (2024)
    target_col = 'target_atingiu_meta_anual_2024'
    counts = df_merged[target_col].value_counts().to_dict()
    print(f"\n[2/5] Target Balanceado '{target_col}':")
    print(f" -> Classe 0 (Abaixo da Meta Anual em 2024): {counts.get(0, 0)} municipios")
    print(f" -> Classe 1 (Atingiu a Meta Anual em 2024): {counts.get(1, 0)} municipios")

    # 3. Pré-processamento e Prevenção de Data Leakage
    print("\n[3/5] Aplicando pre-processamento, imputacao e One-Hot Encoding...")
    X, y = prepare_features_and_target_for_modeling(df_merged, target_col=target_col)
    print(f" -> Matriz de Features X: {X.shape[0]} linhas x {X.shape[1]} variaveis explicativas.")
    print(f" -> Vetor Target y: {len(y)} amostras.")

    # 4. Treinamento da Pipeline de Modelos Supervisionados
    print("\n[4/5] Treinando os 4 algoritmos de classificacao supervisionada...")
    # Usando split estratificado de 25% para teste (10 amostras no teste, 30 no treino)
    pipeline = ClassificationPipeline(test_size=0.25, random_state=42)
    pipeline.fit(X, y)

    # 5. Avaliação e Comparativo de Desempenho
    print("\n[5/5] Gerando metricas de avaliacao e relatorios comparativos...")
    comparison_table = pipeline.evaluate()
    print("\n" + "=" * 60)
    print("RESULTADO COMPARATIVO DOS MODELOS DE CLASSIFICACAO")
    print("=" * 60)
    print(comparison_table.to_string(index=False))
    print("=" * 60)

    # Exibe matrizes de confusão detalhadas
    for model_name in comparison_table['Modelo']:
        cm = pipeline.get_confusion_matrix(model_name)
        print(f"\nMatriz de Confusao ({model_name}):")
        print(f"  Verdadeiros Positivos (TP): {cm['TP']} | Falsos Positivos (FP): {cm['FP']}")
        print(f"  Verdadeiros Negativos (TN): {cm['TN']} | Falsos Negativos (FN): {cm['FN']}")

    # 6. Exportação de Artefatos e Gráficos
    os.makedirs('images', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    chart_path = 'images/model_comparison.png'
    plot_model_comparison(
        comparison_table,
        metric='Acurácia',
        title='Benchmark dos Modelos - Previsao da Meta Anual Educacional',
        save_path=chart_path
    )
    print(f"\n[OK] Grafico comparativo exportado para '{chart_path}'")

    tree_model = pipeline.models.get('Decision Tree')
    if tree_model:
        tree_chart_path = 'images/decision_tree.png'
        plot_decision_tree_structure(
            tree_model,
            feature_names=list(X.columns),
            class_names=['Abaixo Meta', 'Atingiu Meta'],
            save_path=tree_chart_path
        )
        print(f"[OK] Estrutura da Arvore de Decisao exportada para '{tree_chart_path}'")

    model_export_path = 'reports/pipeline_educacional_consolidada.joblib'
    pipeline.export(model_export_path)
    print(f"[OK] Pipeline completa serializada com sucesso em '{model_export_path}'")
    print("\n[SUCESSO] Execucao concluida com 100% de sucesso!")


if __name__ == '__main__':
    main()

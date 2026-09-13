import pytest
import pandas as pd
import numpy as np
import os

from src.preprocessing.data_loader import (
    generate_synthetic_education_data,
    split_features_target,
    load_and_merge_gold_data
)
from src.preprocessing.preprocessor import (
    split_data,
    scale_features,
    prepare_features_and_target_for_modeling
)
from src.modeling.models import get_classification_models, train_all_models, REQUIRES_SCALING
from src.evaluation.evaluator import evaluate_predictions, generate_comparison_table, explain_confusion_matrix
from src.modeling.pipeline import ClassificationPipeline


def test_generate_synthetic_education_data():
    df = generate_synthetic_education_data(n_samples=100, random_state=42)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100, 5)
    assert set(df.columns) == {'taxa_presenca_pct', 'media_proficiencia', 'taxa_alfabetizacao_pct', 'infraestrutura_nota', 'status_aprovacao'}
    assert set(df['status_aprovacao'].unique()).issubset({0, 1})


def test_split_features_target():
    df = generate_synthetic_education_data(n_samples=50, random_state=42)
    X, y = split_features_target(df, target_col='status_aprovacao')
    assert X.shape == (50, 4)
    assert len(y) == 50
    assert 'status_aprovacao' not in X.columns


def test_split_and_scale_data():
    df = generate_synthetic_education_data(n_samples=200, random_state=42)
    X, y = split_features_target(df, 'status_aprovacao')
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    assert X_train.shape == (160, 4)
    assert X_test.shape == (40, 4)
    assert len(y_train) == 160
    assert len(y_test) == 40
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape == (160, 4)
    assert X_test_scaled.shape == (40, 4)
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), np.zeros(4), atol=1e-7)
    np.testing.assert_allclose(X_train_scaled.std(axis=0), np.ones(4), atol=1e-7)


def test_model_training():
    df = generate_synthetic_education_data(n_samples=200, random_state=42)
    X, y = split_features_target(df, 'status_aprovacao')
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    
    models = get_classification_models(random_state=42)
    assert len(models) == 4
    
    trained_models, predictions = train_all_models(
        models=models,
        X_train=X_train,
        X_test=X_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train
    )
    
    assert len(trained_models) == 4
    assert len(predictions) == 4
    for name, pred in predictions.items():
        assert len(pred) == len(y_test)
        assert set(pred).issubset({0, 1})


def test_evaluator():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    
    metrics = evaluate_predictions(y_true, y_pred)
    assert 'accuracy' in metrics
    
    cm_info = explain_confusion_matrix(metrics['confusion_matrix'])
    assert cm_info['TN'] == 3
    assert cm_info['FN'] == 1
    
    preds_dict = {'Model A': np.array([0, 0, 1, 0, 0, 1])}
    table = generate_comparison_table(y_true, preds_dict)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 1


def test_full_classification_pipeline(tmp_path):
    df = generate_synthetic_education_data(n_samples=250, random_state=42)
    X, y = split_features_target(df, 'status_aprovacao')
    
    pipeline = ClassificationPipeline(test_size=0.2, random_state=42)
    pipeline.fit(X, y)
    
    summary = pipeline.evaluate()
    assert isinstance(summary, pd.DataFrame)
    
    X_new = X.iloc[:5]
    preds = pipeline.predict('Decision Tree', X_new)
    assert len(preds) == 5
    
    save_path = os.path.join(tmp_path, "pipeline.joblib")
    pipeline.export(save_path)
    assert os.path.exists(save_path)


def test_load_and_merge_gold_datasets():
    # Tests loading and merging the 3 real datasets in data/
    df_merged = load_and_merge_gold_data(data_dir='data')
    assert isinstance(df_merged, pd.DataFrame)
    assert len(df_merged) == 40
    assert 'target_atingiu_meta_anual_2024' in df_merged.columns
    assert set(df_merged['target_atingiu_meta_anual_2024'].unique()) == {0, 1}
    
    # Preprocessing check
    X, y = prepare_features_and_target_for_modeling(df_merged, target_col='target_atingiu_meta_anual_2024')
    assert len(X) == 40
    assert len(y) == 40
    assert 'target_atingiu_meta_anual_2024' not in X.columns
    assert 'id_municipio' not in X.columns

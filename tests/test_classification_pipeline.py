import pytest
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.preprocessing.data_loader import (
    generate_synthetic_education_data,
    split_features_target,
    load_and_merge_gold_data
)
from src.preprocessing.preprocessor import (
    build_column_transformer,
    extract_raw_features_and_target,
    split_data
)
from src.modeling.models import get_base_classifiers, get_hyperparameter_grids
from src.modeling.pipeline import build_pipeline, NativeClassificationPipeline
from src.modeling.hyperparameter_tuning import tune_all_models
from src.evaluation.evaluator import evaluate_predictions, evaluate_fitted_pipelines, explain_confusion_matrix
from src.evaluation.interpretability import (
    get_feature_names_from_pipeline,
    get_feature_importance_df,
    compute_shap_explanations
)


def test_synthetic_data_generation():
    df = generate_synthetic_education_data(n_samples=50, random_state=42)
    assert len(df) == 50
    assert 'status_aprovacao' in df.columns
    assert set(df['status_aprovacao'].unique()).issubset({0, 1})


def test_column_transformer_creation_and_fit():
    df = generate_synthetic_education_data(n_samples=60, random_state=42)
    X, y = split_features_target(df, 'status_aprovacao')
    numeric_cols = ['taxa_presenca_pct', 'media_proficiencia', 'taxa_alfabetizacao_pct', 'infraestrutura_nota']
    
    col_transformer = build_column_transformer(numeric_features=numeric_cols, categorical_features=[])
    X_trans = col_transformer.fit_transform(X)
    
    assert X_trans.shape == (60, 4)
    # Scaled data should have approx mean 0
    np.testing.assert_allclose(X_trans.mean(axis=0), np.zeros(4), atol=1e-7)


def test_gold_datasets_ingestion_and_leak_free_prep():
    df_merged = load_and_merge_gold_data(data_dir='data')
    assert len(df_merged) == 40
    assert 'target_atingiu_meta_anual_2024' in df_merged.columns

    X, y, num_cols, cat_cols = extract_raw_features_and_target(df_merged, target_col='target_atingiu_meta_anual_2024')
    assert len(X) == 40
    assert len(y) == 40
    assert 'id_municipio' not in X.columns
    assert 'uf_status_meta_2024' not in X.columns
    assert len(num_cols) > 0
    assert len(cat_cols) > 0


def test_native_sklearn_pipeline_fit_predict():
    df_merged = load_and_merge_gold_data(data_dir='data')
    X, y, num_cols, cat_cols = extract_raw_features_and_target(df_merged)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)

    preprocessor = build_column_transformer(num_cols, cat_cols)
    classifiers = get_base_classifiers(random_state=42)
    
    pipe = build_pipeline(preprocessor, classifiers['Logistic Regression'])
    assert isinstance(pipe, Pipeline)

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})


def test_hyperparameter_tuning_gridsearch():
    df_merged = load_and_merge_gold_data(data_dir='data')
    X, y, num_cols, cat_cols = extract_raw_features_and_target(df_merged)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)

    preprocessor = build_column_transformer(num_cols, cat_cols)
    
    # Run tuning with 3 folds for fast test execution
    results_df, best_estimators, best_name, best_pipe = tune_all_models(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        n_splits=3,
        scoring='f1',
        random_state=42
    )

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) >= 4
    assert 'CV_F1' in results_df.columns
    assert best_name in best_estimators
    assert isinstance(best_pipe, Pipeline)


def test_interpretability_and_shap():
    df_merged = load_and_merge_gold_data(data_dir='data')
    X, y, num_cols, cat_cols = extract_raw_features_and_target(df_merged)
    preprocessor = build_column_transformer(num_cols, cat_cols)
    
    clf = get_base_classifiers(random_state=42)['Decision Tree']
    pipe = build_pipeline(preprocessor, clf)
    pipe.fit(X, y)

    # Feature Importance
    df_imp = get_feature_importance_df(pipe)
    assert isinstance(df_imp, pd.DataFrame)
    assert 'Feature' in df_imp.columns
    assert 'Valor' in df_imp.columns
    assert len(df_imp) > 0

    # SHAP Explanations
    explainer, shap_vals, feature_names, X_trans = compute_shap_explanations(pipe, X.iloc[:10])
    assert shap_vals.shape[0] == 10
    assert shap_vals.shape[1] == len(feature_names)


def test_pipeline_container_export_and_reload(tmp_path):
    df_merged = load_and_merge_gold_data(data_dir='data')
    X, y, num_cols, cat_cols = extract_raw_features_and_target(df_merged)
    preprocessor = build_column_transformer(num_cols, cat_cols)

    container = NativeClassificationPipeline(preprocessor)
    clf = get_base_classifiers(random_state=42)['Logistic Regression']
    container.register_classifier('Logistic Regression', clf)
    container.fit_single('Logistic Regression', X, y)

    save_file = os.path.join(tmp_path, "pipeline_test.joblib")
    container.export(save_file)
    assert os.path.exists(save_file)

    loaded = NativeClassificationPipeline.load(save_file)
    preds = loaded.predict('Logistic Regression', X.iloc[:5])
    assert len(preds) == 5

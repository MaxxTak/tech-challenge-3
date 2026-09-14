"""
Data Preprocessing & Native Scikit-Learn ColumnTransformer Module.
Provides native scikit-learn transformers for numeric imputation/scaling
and categorical imputation/one-hot encoding, completely encapsulated to avoid data leakage.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def build_column_transformer(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Constructs an automated, native Scikit-Learn ColumnTransformer:
    - Numeric pipeline: SimpleImputer(median) -> StandardScaler
    - Categorical pipeline: SimpleImputer(constant='NAO_INFORMADO') -> OneHotEncoder(handle_unknown='ignore')
    """
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NAO_INFORMADO')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor


def extract_raw_features_and_target(
    df: pd.DataFrame,
    target_col: str = 'target_atingiu_meta_anual_2024'
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Extracts raw features matrix X and target vector y from the merged Gold dataset.
    Identifies numeric and categorical column names without applying premature pandas transformations.
    Strictly removes future leak columns, metadata, and identifiers.
    """
    df_clean = df.copy()

    if target_col not in df_clean.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # 1. Target vector
    if 'atingiu_meta' in target_col or 'evolucao_positiva' in target_col:
        y = df_clean[target_col].astype(int)
    else:
        y = df_clean[target_col].astype(float)

    # 2. Columns to drop (metadata, IDs, and 2024 future outcome leakage)
    drop_cols = [
        # Keys and metadata
        'id_municipio', 'codigo_uf', 'chave_gold_indicador_municipio',
        'chave_gold_metas_vs_resultados_municipio', 'chave_gold_evolucao_alfabetizacao_uf',
        '_ingestion_date', '_execution_id', 'ano', 'serie', 'rede', 'rede_resultado', 'rede_meta',
        # Target columns and 2024 outcome leakage
        'target_atingiu_meta_anual_2024', 'target_evolucao_positiva_2024', 'target',
        'target_resultado_alfabetizacao_2024_pct',
        'uf_status_meta_2024', 'uf_variacao_pp_2024', 'status_meta_ano', 'status_meta_2030',
        # High nullity sample-level columns (no predictive variance)
        'qtd_alunos_amostra', 'qtd_alunos_alfabetizados_amostra', 'qtd_alunos_presentes_amostra',
        'taxa_alfabetizacao_alunos_amostra_pct', 'taxa_presenca_alunos_amostra_pct',
        'media_proficiencia_alunos_amostra', 'soma_peso_alunos_amostra',
        'participacao_municipio_pct', 'meta_alfabetizacao_2024_pct', 'meta_alfabetizacao_2030_pct',
        'meta_ano_resultado_pct', 'diferenca_meta_ano_pp'
    ]

    X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])

    # 3. Clean string numeric formatting on media_portugues if present
    if 'media_portugues' in X.columns and X['media_portugues'].dtype == object:
        X['media_portugues'] = (
            X['media_portugues']
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        X['media_portugues'] = pd.to_numeric(X['media_portugues'], errors='coerce')

    # Ensure boolean flags are treated as integer
    if 'flag_possui_distribuicao_niveis' in X.columns:
        X['flag_possui_distribuicao_niveis'] = X['flag_possui_distribuicao_niveis'].astype(int)

    # 4. Separate column types for ColumnTransformer
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    return X, y, numeric_features, categorical_features


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train-test split executed BEFORE any preprocessing,
    guaranteeing zero data leakage during evaluation.
    """
    stratify_target = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

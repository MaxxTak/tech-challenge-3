"""
Data Preprocessing: Train-Test Split, Feature Scaling, and Dataset Cleaning.
Implements the rule established in the supervised classification blueprint:
- Scaled data is required for distance-based/gradient models (Logistic Regression, SVM).
- Raw unscaled data is used for tree-based & probabilistic models (Decision Tree, Naive Bayes).
- Zero data leakage between train and test splits, and across prediction years.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits feature matrix X and target vector y into train and test sets.
    Uses stratification by default to preserve target class proportions.
    """
    stratify_target = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardizes features using z-score normalization: (x - u) / s.
    CRITICAL: Fits ONLY on X_train to prevent data leakage, then transforms X_test.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_features_and_target_for_modeling(
    df: pd.DataFrame,
    target_col: str = 'target_atingiu_meta_anual_2024'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the merged multi-dataset educational dataframe for machine learning:
    1. Isolates the target vector y.
    2. Drops non-feature columns: IDs, metadata, keys, and future 2024 indicators (preventing leakage).
    3. Drops uninformative/zero-variance sample columns.
    4. Imputes categorical features and numerical features appropriately.
    5. One-Hot Encodes categorical variables.
    6. Returns clean feature matrix X and target Series y.
    """
    df_clean = df.copy()
    
    if target_col not in df_clean.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    
    # 1. Extract target vector
    y = df_clean[target_col].astype(int)
    
    # 2. Define columns to drop (Metadata, Identifiers, and Target Leakage)
    leak_and_meta_cols = [
        # IDs and keys
        'id_municipio', 'codigo_uf', 'chave_gold_indicador_municipio',
        'chave_gold_metas_vs_resultados_municipio', 'chave_gold_evolucao_alfabetizacao_uf',
        '_ingestion_date', '_execution_id', 'ano', 'serie', 'rede', 'rede_resultado', 'rede_meta',
        # Other potential targets or future leak columns
        'target_atingiu_meta_anual_2024', 'target_evolucao_positiva_2024', 'target',
        'uf_status_meta_2024', 'uf_variacao_pp_2024', 'status_meta_ano', 'status_meta_2030',
        # Sparsely populated or zero-variance sample-level columns
        'qtd_alunos_amostra', 'qtd_alunos_alfabetizados_amostra', 'qtd_alunos_presentes_amostra',
        'taxa_alfabetizacao_alunos_amostra_pct', 'taxa_presenca_alunos_amostra_pct',
        'media_proficiencia_alunos_amostra', 'soma_peso_alunos_amostra',
        # Future targets / planning indicators with high nullity
        'participacao_municipio_pct', 'meta_alfabetizacao_2024_pct', 'meta_alfabetizacao_2030_pct',
        'meta_ano_resultado_pct', 'diferenca_meta_ano_pp'
    ]
    
    X = df_clean.drop(columns=[c for c in leak_and_meta_cols if c in df_clean.columns])
    
    # 3. Clean and fix numeric formatting if any remain
    if 'media_portugues' in X.columns and X['media_portugues'].dtype == object:
        X['media_portugues'] = (
            X['media_portugues']
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        X['media_portugues'] = pd.to_numeric(X['media_portugues'], errors='coerce')
        
    # 4. Impute Categorical missing values
    if 'faixa_prioridade_intervencao' in X.columns:
        X['faixa_prioridade_intervencao'] = X['faixa_prioridade_intervencao'].fillna('NAO_INFORMADO')
        
    # 5. Impute remaining numeric missing values with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            
    # 6. One-Hot Encoding for categorical features
    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
    # Ensure boolean flags are binary integers
    bool_cols = X.select_dtypes(include=['bool']).columns
    for c in bool_cols:
        X[c] = X[c].astype(int)
        
    return X, y


def clean_education_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy helper for single-table educational dataframe cleaning.
    """
    df_clean = df.copy()
    
    cols_to_clean = ['media_portugues', 'media_proficiencia_alunos_amostra', 'soma_peso_alunos_amostra']
    for c in cols_to_clean:
        if c in df_clean.columns:
            df_clean[c] = (
                df_clean[c]
                .astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .astype(float)
            )
            
    df_clean.fillna(0, inplace=True)
    
    sample_cols = [
        'qtd_alunos_amostra', 'qtd_alunos_alfabetizados_amostra', 'qtd_alunos_presentes_amostra',
        'taxa_alfabetizacao_alunos_amostra_pct', 'taxa_presenca_alunos_amostra_pct',
        'media_proficiencia_alunos_amostra', 'soma_peso_alunos_amostra'
    ]
    df_clean = df_clean.drop(columns=[c for c in sample_cols if c in df_clean.columns])
    
    meta_cols = [
        'id_municipio', 'codigo_uf', 'chave_gold_indicador_municipio',
        '_ingestion_date', '_execution_id', 'ano', 'serie', 'rede'
    ]
    df_clean = df_clean.drop(columns=[c for c in meta_cols if c in df_clean.columns])
    
    if 'status_meta_2030' in df_clean.columns:
        df_clean['target'] = df_clean['status_meta_2030'].apply(
            lambda x: 1 if x == 'ACIMA_OU_IGUAL_META_2030' else 0
        )
        df_clean = df_clean.drop(columns=['status_meta_2030'])
        
    categorical_cols = df_clean.select_dtypes(include=['object', 'string']).columns
    if len(categorical_cols) > 0:
        df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
        
    bool_cols = df_clean.select_dtypes(include=['bool']).columns
    for c in bool_cols:
        df_clean[c] = df_clean[c].astype(int)
            
    return df_clean

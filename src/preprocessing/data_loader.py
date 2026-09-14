"""
Data Ingestion and Dataset Merging for Educational Quality Pipeline.
Handles individual loading as well as multi-dataset joins across municipal
indicators, intervention priorities, and state/national evolutionary benchmarks.
"""

from typing import Tuple, Dict, Any, Optional
import os
import glob
import numpy as np
import pandas as pd


def generate_synthetic_education_data(
    n_samples: int = 200,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a synthetic education classification dataset reflecting educational
    quality and student approval rates for automated testing.
    """
    np.random.seed(random_state)
    
    taxa_presenca_pct = np.random.uniform(50.0, 100.0, n_samples)
    media_proficiencia = np.random.uniform(0.0, 10.0, n_samples)
    taxa_alfabetizacao_pct = np.random.uniform(40.0, 100.0, n_samples)
    infraestrutura_nota = np.random.randint(1, 6, n_samples)
    
    # Ground truth business logic rule
    status_aprovacao = (
        (taxa_presenca_pct > 75.0) &
        (media_proficiencia > 5.0)
    ).astype(int)
    
    return pd.DataFrame({
        'taxa_presenca_pct': taxa_presenca_pct.round(2),
        'media_proficiencia': media_proficiencia.round(2),
        'taxa_alfabetizacao_pct': taxa_alfabetizacao_pct.round(2),
        'infraestrutura_nota': infraestrutura_nota,
        'status_aprovacao': status_aprovacao
    })


def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads a CSV or Excel dataset into a pandas DataFrame."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def split_features_target(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separates features matrix X and target vector y."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def load_gold_datasets(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all 3 Gold layer datasets from the data directory:
    1. df_ind: Municipal quality indicators (2023)
    2. df_metas: Municipal goals and intervention priorities (2023)
    3. df_uf: State temporal evolution and targets (2023-2024)
    """
    ind_path = os.path.join(data_dir, 'tech-challenges-results-sql-result_gold_indicador_municipio.csv')
    metas_path = os.path.join(data_dir, 'tech-challenges-results-sql-result_gold_metas_vs_resultados_municipio.csv')
    uf_path = os.path.join(data_dir, 'tech-challenges-results-sql-result_gold_evolucao_alfabetizacao_uf.csv')
    
    df_ind = load_dataset(ind_path)
    df_metas = load_dataset(metas_path)
    df_uf = load_dataset(uf_path)
    
    return df_ind, df_metas, df_uf


def merge_gold_datasets(
    df_ind: pd.DataFrame,
    df_metas: pd.DataFrame,
    df_uf: pd.DataFrame
) -> pd.DataFrame:
    """
    Unifies the 3 Gold layer datasets into a single consolidated analytical dataset:
    - Combines municipal indicators with intervention priority tiers (on id_municipio).
    - Enriches with state (UF) and national macro-context from 2023 (on sigla_uf).
    - Crosses with 2024 state-level outcome to establish a balanced target:
      'target_atingiu_meta_anual_2024' (1 = ATINGIU_META_ANUAL, 0 = ABAIXO_META_ANUAL).
    """
    df_ind_clean = df_ind.copy()
    df_metas_clean = df_metas.copy()
    
    # 1. Standardize formatting on numeric string columns
    for df in [df_ind_clean, df_metas_clean]:
        if 'media_portugues' in df.columns:
            df['media_portugues'] = (
                df['media_portugues']
                .astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
            )
            df['media_portugues'] = pd.to_numeric(df['media_portugues'], errors='coerce')
    
    # 2. Select relevant municipal priority and planning features from df_metas
    metas_features = [
        'id_municipio',
        'faixa_prioridade_intervencao',
        'percentual_participacao_pct',
        'meta_alfabetizacao_2024_pct',
        'meta_alfabetizacao_2030_pct'
    ]
    metas_subset = df_metas_clean[[c for c in metas_features if c in df_metas_clean.columns]].copy()
    if 'percentual_participacao_pct' in metas_subset.columns:
        metas_subset = metas_subset.rename(columns={'percentual_participacao_pct': 'participacao_municipio_pct'})
        
    df_mun = pd.merge(df_ind_clean, metas_subset, on='id_municipio', how='left')
    
    # 3. Add 2023 State Context features (No Data Leakage)
    uf_2023 = df_uf[df_uf['ano'] == 2023].copy()
    context_cols = {
        'resultado_alfabetizacao_uf_pct': 'uf_taxa_alfabetizacao_2023',
        'ranking_uf_no_ano': 'uf_ranking_2023',
        'taxa_alfabetizacao_brasil_pct': 'taxa_alfabetizacao_brasil_2023_pct',
        'diferenca_brasil_pp': 'uf_diferenca_brasil_2023_pp',
        'percentual_participacao_uf_pct': 'uf_participacao_2023_pct',
        'percentual_participacao_brasil_pct': 'participacao_brasil_2023_pct'
    }
    cols_to_extract = ['sigla_uf'] + [c for c in context_cols.keys() if c in uf_2023.columns]
    uf_2023_sub = uf_2023[cols_to_extract].rename(columns=context_cols)
    
    df_enriched = pd.merge(df_mun, uf_2023_sub, on='sigla_uf', how='left')
    
    # 4. Cross with 2024 State Outcomes (Ground Truth Target)
    uf_2024 = df_uf[df_uf['ano'] == 2024].copy()
    if 'status_meta_ano' in uf_2024.columns:
        uf_2024['target_atingiu_meta_anual_2024'] = (
            uf_2024['status_meta_ano'] == 'ATINGIU_META_ANUAL'
        ).astype(int)
    if 'variacao_alfabetizacao_uf_pp' in uf_2024.columns:
        uf_2024['target_evolucao_positiva_2024'] = (
            uf_2024['variacao_alfabetizacao_uf_pp'] > 0
        ).astype(int)
        
    target_cols = ['sigla_uf']
    if 'target_atingiu_meta_anual_2024' in uf_2024.columns:
        target_cols.append('target_atingiu_meta_anual_2024')
    if 'target_evolucao_positiva_2024' in uf_2024.columns:
        target_cols.append('target_evolucao_positiva_2024')
    if 'resultado_alfabetizacao_uf_pct' in uf_2024.columns:
        uf_2024['target_resultado_alfabetizacao_2024_pct'] = uf_2024['resultado_alfabetizacao_uf_pct']
        target_cols.append('target_resultado_alfabetizacao_2024_pct')
        
    df_final = pd.merge(df_enriched, uf_2024[target_cols], on='sigla_uf', how='left')
    return df_final


def load_and_merge_gold_data(data_dir: str = 'data') -> pd.DataFrame:
    """One-stop helper to load and merge all 3 Gold datasets."""
    df_ind, df_metas, df_uf = load_gold_datasets(data_dir=data_dir)
    return merge_gold_datasets(df_ind, df_metas, df_uf)

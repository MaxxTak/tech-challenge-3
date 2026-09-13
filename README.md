# Tech Challenge - Fase 3: Machine Learning Supervisionado para Alfabetização Infantil

Este repositório contém a solução completa de Ciência de Dados e Engenharia de Machine Learning para a **Fase 3 do Tech Challenge (Pós-Tech Data Analytics & AI)**.

O projeto implementa uma esteira automatizada em **Scikit-Learn nativo**, unificando as tabelas processadas na camada **Gold** da Fase 2 para prever o risco e a probabilidade de cumprimento das metas educacionais municipais de alfabetização infantil, fornecendo subsídios preditivos para intervenções de políticas públicas.

---

## 1. Visão Geral & Contexto de Negócio

* **Desafio**: Atrasos na identificação de redes municipais e escolas com baixa proficiência em leitura e escrita comprometem o futuro acadêmico das crianças. A atuação reativa baseada em relatórios anuais tardios impede o reforço pedagógico preventivo.
* **Objetivo de Machine Learning**: Construir um classificador supervisionado de alta performance e explicabilidade, capaz de antecipar com 1 ano de antecedência se a rede pública municipal atingirá a meta anual de alfabetização infantil.
* **Fontes de Dados (Camada Gold)**:
  1. `gold_indicador_municipio.csv`: Indicadores de alfabetização, proficiência média em português e dados amostrais locais (2023).
  2. `gold_metas_vs_resultados_municipio.csv`: Planejamento plurianual de metas e classificação por faixas de prioridade de intervenção pedagógica (`ALTA`, `MEDIA`, `BAIXA`).
  3. `gold_evolucao_alfabetizacao_uf.csv`: Série histórica (2023–2024), benchmarks nacionais e metas anuais estaduais consolidadas.

---

## 2. Destaques da Arquitetura Técnica (100% de Conformidade)

* **Pipeline Scikit-Learn Nativo**:
  - `ColumnTransformer` desacoplado:
    - **Variáveis Numéricas**: `SimpleImputer(strategy='median')` + `StandardScaler()`.
    - **Variáveis Categóricas**: `SimpleImputer(strategy='constant', fill_value='NAO_INFORMADO')` + `OneHotEncoder(handle_unknown='ignore')`.
  - Encapsulamento estrito: `Pipeline([('preprocessor', col_transformer), ('classifier', estimator)])`.
* **Zero Data Leakage**:
  - Variáveis de desfecho do ciclo seguinte (2024) rigorosamente isoladas no vetor target $y$.
  - Transformadores ajustados (*fit*) exclusivamente sobre os dados de treino em cada partição.
* **Otimização de Hiperparâmetros & Validação Estatística**:
  - `StratifiedKFold(n_splits=5, shuffle=True)` para preservar a proporção das classes em todos os folds.
  - `GridSearchCV` sistemático em 5 famílias de algoritmos: **Logistic Regression, Decision Tree, SVM, Naive Bayes e Random Forest**.
* **Interpretabilidade Avançada**:
  - Cálculo de **SHAP Values** (`shap.TreeExplainer` e `shap.LinearExplainer`).
  - Ranking de **Global Feature Importance** baseado em MDI e magnitude de coeficientes.
  - Visualização de regras da **Árvore de Decisão** (`plot_tree`).

---

## 3. Estrutura Modular do Repositório

```text
tech-challenge-fase3/
│
├── 📁 data/                                  # Datasets da camada Gold (SQL outputs)
│   ├── tech-challenges-results-sql-result_gold_indicador_municipio.csv
│   ├── tech-challenges-results-sql-result_gold_metas_vs_resultados_municipio.csv
│   └── tech-challenges-results-sql-result_gold_evolucao_alfabetizacao_uf.csv
│
├── 📁 notebooks/                             # Notebooks interativos para entrega acadêmica
│   └── 01_eda_and_supervised_pipeline.ipynb # EDA, execução da pipeline, SHAP e conclusões
│
├── 📁 src/                                   # Código-fonte modular em Python
│   ├── __init__.py
│   ├── 📁 preprocessing/                     # Ingestão e transformadores Scikit-Learn
│   │   ├── __init__.py
│   │   ├── data_loader.py                   # Ingestão e cruzamento das 3 bases Gold
│   │   └── preprocessor.py                  # Definição do ColumnTransformer nativo e split
│   │
│   ├── 📁 modeling/                          # Construção de pipelines e Grid Search
│   │   ├── __init__.py
│   │   ├── models.py                        # Fábrica de classificadores e grids de hiperparâmetros
│   │   ├── pipeline.py                      # Fábrica de Pipeline e container de serialização
│   │   └── hyperparameter_tuning.py         # Orquestrador de GridSearchCV e StratifiedKFold
│   │
│   ├── 📁 evaluation/                        # Avaliação estatística e interpretabilidade
│   │   ├── __init__.py
│   │   ├── evaluator.py                     # Métricas (F1, ROC-AUC, Matriz de Confusão)
│   │   └── interpretability.py              # Cálculo de SHAP Values e Feature Importance
│   │
│   └── 📁 visualization/                     # Geração e salvamento de gráficos diagnósticos
│       ├── __init__.py
│       └── plots.py                         # Curvas, matrizes, Feature Importance e SHAP plots
│
├── 📁 reports/                               # Documentações técnicas e artefatos serializados
│   ├── TECHNICAL_SPECIFICATION.md           # Especificação técnica formal de arquitetura
│   └── pipeline_educacional_consolidada.joblib # Pipeline campeã serializada para produção
│
├── 📁 images/                                # Gráficos exportados automaticamente
│   ├── model_comparison.png                 # Benchmark comparativo de performance
│   ├── confusion_matrix_best_model.png       # Matriz de confusão do melhor modelo
│   ├── feature_importance.png               # Top 10 variáveis mais preditivas
│   ├── shap_summary.png                     # Gráfico de impacto SHAP Beeswarm
│   └── decision_tree.png                    # Regras hierárquicas da árvore de decisão
│
├── 📁 tests/                                 # Suíte automatizada de testes unitários
│   ├── __init__.py
│   └── test_classification_pipeline.py      # Testes de ingestão, ColumnTransformer, Grid e SHAP
│
├── main.py                                  # Script executivo principal (orquestrador)
├── requirements.txt                          # Dependências do ambiente com versões fixadas
├── README.md                                 # Esta documentação
└── .gitignore                               # Exclusões de arquivos temporários e dados sensíveis
```

---

## 4. Resultados do Benchmark e Performance

A esteira foi avaliada através de validação cruzada estratificada em 5 folds e testada em um conjunto de teste cego (25% das amostras):

| Modelo | F1-Score (Validação Cruzada) | Acurácia (Teste) | Precisão (Teste) | Recall (Teste) | F1-Score (Teste) | ROC-AUC (Teste) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **0.9429** | **90,0%** | **100,0%** | **75,0%** | **0,8571** | **0.9167** |
| **Naive Bayes** | 0.8743 | 90,0% | 100,0% | 75,0% | 0,8571 | 0.8750 |
| **Logistic Regression** | 0.8548 | 90,0% | 100,0% | 75,0% | 0,8571 | 0.8750 |
| **Decision Tree** | 0.7981 | 90,0% | 100,0% | 75,0% | 0,8571 | **0.9792** |
| **Random Forest** | 0.8548 | 80,0% | 100,0% | 50,0% | 0,6667 | 0.9583 |

### Matriz de Confusão no Teste (SVM / Decision Tree / Logistic Regression):
* **Verdadeiros Negativos (TN)**: 6 (Municípios que não atingiriam a meta identificados com exatidão)
* **Falsos Positivos (FP)**: 0 (**Zero falso otimismo!**)
* **Falsos Negativos (FN)**: 1
* **Verdadeiros Positivos (TP)**: 3 (Identificação precisa dos ecossistemas em rota de sucesso)

---

## 5. Como Instalar e Executar

### 5.1 Configuração do Ambiente Virtual
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Ativar ambiente (Linux/Mac)
source .venv/bin/activate

# Instalar dependências exatas
pip install -r requirements.txt
```

### 5.2 Execução da Pipeline Completa
```bash
python main.py
```
O script executa automaticamente a ingestão, o ColumnTransformer, o GridSearchCV em 5 folds, a avaliação estatística, os cálculos de SHAP e a geração dos gráficos em `images/`.

### 5.3 Execução dos Testes Automatizados
```bash
pytest tests/ -v
```

---

## 6. Valor Estratégico para Políticas Públicas

1. **Previsibilidade Antecipada**: Permite que secretarias municipais e estaduais de educação intervenham pedagogicamente com 1 ano de antecedência.
2. **Priorização Eficiente de Recursos**: Municípios mapeados com alta prioridade de intervenção e baixa probabilidade de meta recebem alocação direta de reforço de alfabetização.
3. **Decisões Transparentes**: Com os valores de **SHAP** e a **Árvore de Decisão**, gestores públicos entendem exatamente por que determinado município está sob alerta de risco.
# Tech Challenge - Fase 3

Template padronizado alinhado a todas as seções e especificações técnicas exigidas para o projeto.

### **Visão Geral do Projeto**
* **Contexto & Objetivos de Negócio**: Descreve os objetivos de alfabetização infantil, fontes de dados da camada Gold e bases de enriquecimento externo (IBGE, Censo Escolar, PNAD).
* **Arquitetura & Estrutura**: Detalha a estrutura modular de diretórios (`src/preprocessing`, `src/modeling`, `src/evaluation`, `src/visualization`) juntamente com os arquivos de configuração necessários.
* **Metodologia Técnica**: Resume pipelines automatizados do Scikit-learn, controle de vazamento de dados (*data leakage*), estratégias de validação cruzada, matrizes de avaliação de modelos e interpretabilidade via SHAP.
* **Valor Estratégico**: Destaca aplicações em políticas públicas para avaliação de risco municipal e desdobramentos futuros no roadmap.


### **Estrutura Mínima do Repositório**

```
tech-challenge-fase3/
│
├── 📁 data/                  # Datasets da camada Gold e arquivos de enriquecimento externo
├── 📁 notebooks/             # Análise Exploratória de Dados (EDA) e notebooks experimentais
├── 📁 src/                   # Código-fonte modular em Python
│   ├── 📁 preprocessing/     # Transformadores customizados, imputação e codificação de variáveis
│   ├── 📁 modeling/          # Definições de pipeline, treinamento e lógica de grid search
│   ├── 📁 evaluation/        # Métricas de modelos, valores SHAP e importância de atributos
│   └── 📁 visualization/     # Funções personalizadas para geração de gráficos e visualizações
│
├── 📁 reports/               # Documentação técnica, notas de políticas públicas e resumos executivos
├── 📁 images/                # Gráficos e figuras exportados para o README e relatórios
├── requirements.txt          # Dependências do ambiente e versões exatas das bibliotecas
├── README.md                 # Documentação técnica e executiva do projeto
└── .gitignore                # Exclusões do Git (arquivos de dados, cache, ambientes virtuais)
```


### **Detalhamento dos Componentes**

#### **1. Módulos Principais de Código (`src/`)**

* **`src/preprocessing/`**: Contém transformadores personalizados do Scikit-learn, imputadores numéricos e codificadores categóricos.
* **`src/modeling/`**: Contém scripts para construir objetos `Pipeline` de ponta a ponta, executar `GridSearchCV` e salvar artefatos dos modelos treinados.
* **`src/evaluation/`**: Implementa métricas de desempenho (F1-score, ROC-AUC), verificações de validação cruzada e ferramentas de interpretabilidade de modelos (SHAP e *Feature Importance*).
* **`src/visualization/`**: Funções para gerar gráficos de EDA, matrizes de confusão, curvas ROC e gráficos de resumo do SHAP (*summary plots*).

#### **2. Documentação & Artefatos**

* **`notebooks/`**: Dedicado à exploração inicial de dados, testes de hipóteses e prototipagem interativa.
* **`reports/`**: Contém análises estratégicas de políticas públicas e resumos técnicos sobre o risco de alfabetização nos municípios.
* **`images/`**: Armazena elementos visuais estáticos utilizados para enriquecer o `README.md` e a documentação dos relatórios.
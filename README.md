# pgrad_pucrio_datascience2
MVP  Machine Learning &amp; Analytics - PUCRio
# MVP — Previsão de Churn de Clientes (Telco Customer Churn)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielcsjunqueira/pgrad_pucrio_datascience/blob/main/Pos_Graduacao_PUCRio_Sprint_Machine_Learning_&_Analytics_Daniel_Junqueira.ipynb)

MVP de Machine Learning & Analytics desenvolvido para a Pós-Graduação em Ciência de Dados e Analytics da PUC-Rio, com o objetivo de prever o cancelamento de clientes (**churn**) de uma empresa de telecomunicações a partir de dados demográficos, de serviços contratados e financeiros.

> **Autor:** Daniel Cardoso e Silva Junqueira
> **Curso:** Pós-Graduação em Ciência de Dados e Analytics — PUC-Rio
> **Disciplina:** Machine Learning & Analytics

---

## Sobre o projeto

Este repositório contém o MVP completo de um problema de **classificação binária supervisionada**: prever se um cliente irá cancelar (`Churn = Yes`) ou permanecer ativo (`Churn = No`), com base em suas características e histórico de consumo.

O projeto dá continuidade a uma análise exploratória feita em uma sprint anterior (Análise de Dados e Boas Práticas) sobre o mesmo dataset, avançando agora para a construção de um pipeline completo de Machine Learning: preparação dos dados, definição de baseline, treinamento e comparação de modelos, otimização de hiperparâmetros, avaliação crítica dos resultados e execução de extensões adicionais (rebalanceamento de classes, engenharia de atributos, modelos de boosting avançados e interpretabilidade).

## Objetivo

Construir e avaliar modelos de Machine Learning capazes de identificar antecipadamente clientes com alto risco de cancelamento, apoiando ações estratégicas de retenção. O foco do trabalho não é obter o melhor modelo possível a qualquer custo, mas construir uma solução **coerente, reprodutível, bem documentada e tecnicamente justificada**.

## Dataset

- **Nome:** Telco Customer Churn
- **Fonte original:** [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Registros:** 7.043 clientes
- **Atributos:** 20 variáveis preditoras (demográficas, serviços contratados, contrato, forma de pagamento e valores cobrados) + 1 variável-alvo (`Churn`)
- **Carga no notebook:** diretamente via URL *raw* deste repositório, sem necessidade de upload manual, login ou chave de API:

```python
URL = "https://raw.githubusercontent.com/danielcsjunqueira/pgrad_pucrio_datascience/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(URL)
```

## Estrutura do repositório

```
pgrad_pucrio_datascience/
├── Pos_Graduacao_PUCRio_Sprint_Machine_Learning_&_Analytics_Daniel_Junqueira.ipynb   # Notebook principal do MVP
├── WA_Fn-UseC_-Telco-Customer-Churn.csv           # Dataset utilizado
└── README.md                                      # Este arquivo
```

## Como executar

1. Clique no badge **"Open in Colab"** no topo deste README (ou acesse diretamente pelo link do notebook acima).
2. No menu do Colab, selecione **Ambiente de execução → Executar tudo**.
3. Não é necessário nenhum upload manual, login ou configuração adicional — o dataset é carregado automaticamente a partir deste repositório.

Alternativamente, para executar localmente:

```bash
git clone https://github.com/danielcsjunqueira/pgrad_pucrio_datascience.git
cd pgrad_pucrio_datascience
pip install -r requirements.txt   # ou instale as bibliotecas listadas abaixo
jupyter notebook MVP_ML_Analytics_Churn_DanielJunqueira.ipynb
```

## Metodologia (estrutura do notebook)

| Seção | Conteúdo |
|---|---|
| 1 | Definição do problema, objetivo, hipóteses e critérios de sucesso |
| 2 | Ambiente, bibliotecas e reprodutibilidade (seed fixa) |
| 3 | Carga dos dados e tratamento de qualidade (`TotalCharges`) |
| 4 | Análise exploratória (target, distribuições, correlações, categóricas x churn) |
| 5 | Preparação dos dados, divisão treino/teste e pipeline de pré-processamento |
| 6 | Baseline (`DummyClassifier`) e modelos candidatos (Regressão Logística, Random Forest, Gradient Boosting, KNN) |
| 7 | Treinamento e avaliação inicial via validação cruzada |
| 8 | Otimização de hiperparâmetros (`RandomizedSearchCV`) |
| 9 | Avaliação final no conjunto de teste, matriz de confusão, overfitting/underfitting e análise de erros |
| 10 | Comparação final dos modelos |
| 11 | Boas práticas e rastreabilidade das decisões |
| 12 | **Extensões:** SMOTE + ajuste de limiar, engenharia de atributos, XGBoost/LightGBM + SHAP |
| 13 | Conclusão |
| 14 | Salvamento de artefatos |

## Principais resultados

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting (otimizado) + NumServicosContratados | 0,8062 | 0,6760 | 0,5187 | 0,5870 | **0,8471** |
| Gradient Boosting (otimizado) — pipeline principal | 0,8062 | 0,6747 | 0,5214 | 0,5882 | 0,8470 |
| Gradient Boosting (otimizado) + SMOTE | 0,7743 | 0,5576 | 0,7246 | **0,6302** | 0,8433 |
| XGBoost | 0,7566 | 0,5281 | 0,7781 | 0,6246 | 0,8394 |
| LightGBM | 0,7637 | 0,5400 | 0,7406 | 0,6246 | 0,8350 |

**Principais achados:**
- O **Gradient Boosting otimizado** é o modelo com melhor capacidade de ranquear clientes por risco (maior ROC-AUC), com ou sem a engenharia de atributos adicional.
- A criação do atributo `NumServicosContratados` teve efeito praticamente nulo, indicando que essa informação já era capturada pelas variáveis categóricas originais.
- O **SMOTE** trouxe o melhor F1-score, ao custo de recall mais alto e precisão mais baixa — um efeito de deslocamento do limiar de decisão, mais do que de melhoria na discriminação do modelo (o ROC-AUC quase não se altera).
- **XGBoost** e **LightGBM** priorizaram ainda mais o recall, mas não superaram o Gradient Boosting nativo do scikit-learn em nenhuma métrica relevante neste dataset.

*(Consulte a Seção 12.5 e a Conclusão do notebook para a análise completa, incluindo a interpretabilidade via SHAP.)*

## Principais decisões técnicas

- Remoção do identificador `customerID` das features (sem poder preditivo real).
- Tratamento do problema de qualidade de dados em `TotalCharges` (imputação coerente com clientes de `tenure = 0`).
- Codificação categórica via One-Hot Encoding (em vez de Label Encoding), por se tratar majoritariamente de variáveis nominais.
- Pipeline de pré-processamento (`ColumnTransformer`) ajustado exclusivamente com dados de treino, evitando vazamento de dados em todas as etapas (validação cruzada, otimização de hiperparâmetros e SMOTE).
- ROC-AUC como métrica principal de seleção, por robustez ao desbalanceamento de classes (~27% churn).

## Limitações e próximos passos

- Dataset estático, sem dimensão temporal — a análise de robustez a mudanças no perfil dos dados (Seção 12.4) é um *proxy*, não uma validação temporal real.
- Custos de negócio usados no ajuste de limiar são hipotéticos e deveriam ser calibrados com valores reais da área de negócio.
- Próximos passos sugeridos: calibração de custos reais, otimização de hiperparâmetros de XGBoost/LightGBM, estudo com novos atributos derivados.

Detalhes completos na Seção 13 (Conclusão) do notebook.

## Tecnologias utilizadas

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost, LightGBM
- SHAP

## Licença

O dataset utilizado é de uso público, disponibilizado no Kaggle para fins educacionais e de pesquisa. Este repositório é disponibilizado para fins acadêmicos, como parte da avaliação do MVP da Pós-Graduação em Ciência de Dados e Analytics da PUC-Rio.

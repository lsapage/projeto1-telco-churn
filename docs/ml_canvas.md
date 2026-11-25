# ML Canvas: previsão de churn (Telco)

Documento da fase de *Business Understanding* (CRISP-DM). Define o problema, quem está envolvido, que decisão o modelo apoia e como mediremos o sucesso, **antes** da modelagem.

## 1. Proposta de valor
- **Problema:** a operadora está perdendo clientes em ritmo acelerado (26,5% da base cancelou no trimestre) e não sabe em quem concentrar os esforços de retenção.
- **Solução:** um modelo que estima, para cada cliente, a **probabilidade de cancelamento** e ordena a base do maior para o menor risco.
- **Para quem:** time de retenção / CRM, que passa a trabalhar com uma lista priorizada em vez de abordar clientes ao acaso.
- **Ganho esperado:** com a mesma capacidade operacional, alcançar uma parcela maior da receita que seria perdida com cancelamentos.

## 2. Stakeholders
| Papel | Quem | Interesse / envolvimento |
|---|---|---|
| Sponsor executivo | Diretoria comercial / de clientes | Reduzir a perda de receita por churn; aprova o projeto |
| Usuário da previsão | Time de retenção / CRM | Recebe a lista priorizada e executa as campanhas |
| Especialistas de domínio | Marketing, atendimento | Validam as variáveis e definem as ofertas de retenção |
| Donos dos dados | TI / engenharia de dados, faturamento | Fornecem e atualizam cadastro, contratos e cobrança |
| Conformidade | Jurídico / DPO | Garantem uso dos dados conforme a LGPD |
| Time técnico | Cientista / engenheiro de ML | Constrói, valida, publica e monitora o modelo |

## 3. Tarefa de ML
- **Tipo:** classificação binária supervisionada.
- **Alvo:** `Churn Value` (1 = cancelou no trimestre; 0 = permaneceu).
- **Saída do modelo:** probabilidade de cancelamento (`predict_proba`), usada para **ordenar** os clientes, e a classe 0/1 (`predict`) para um limiar escolhido.
- **Linha de base:** `DummyClassifier` (piso) e Regressão Logística.

## 4. Decisões apoiadas pelo modelo
- **Capacidade de abordagem:** a operação consegue trabalhar **20% da base** por ciclo (~1.400 clientes), próximo da taxa de churn observada (26,5%).
- **Regra de decisão:** os 20% de maior probabilidade entram na campanha de retenção.
- **Ação:** definida pelo negócio (e-mail, contato ativo, oferta, desconto). O modelo entrega **quem** priorizar; **como** abordar fica com o time de retenção.
- **Frequência:** previsão em lote, a cada ciclo de campanha (ex.: mensal). Previsão individual sob demanda via API.

## 5. Dados
- **Fonte:** IBM Telco Customer Churn, versão estendida: 7.043 clientes × 33 variáveis (Califórnia, EUA), em `data/raw/`.
- **Coleta em produção (hipótese):** cadastro, contratos e faturamento consolidados ao fim de cada trimestre, com o rótulo de churn observado no trimestre seguinte.


## 6. Features
| Grupo | Variáveis |
|---|---|
| Demográficas | `Gender`, `Senior Citizen`, `Partner`, `Dependents` |
| Relacionamento | `Tenure Months`, `Contract`, `Paperless Billing`, `Payment Method` |
| Serviços | `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies` |
| Financeiras | `Monthly Charges`, `Total Charges` |
| Geográficas (a decidir na EDA) | `City`, `Zip Code`, `Latitude`, `Longitude` |

**Excluídas:**
| Variável | Motivo |
|---|---|
| `Churn Label`, `Churn Reason` | Vazamento: são o próprio alvo ou consequência dele |
| `Churn Score` | Saída de outro modelo (IBM SPSS); usado só como **benchmark externo** |
| `CLTV` | Estimativa corporativa, não comportamento do cliente |
| `CustomerID`, `Count`, `Country`, `State`, `Lat Long` | Identificador, constantes ou redundantes |

## 7. Avaliação offline
**Por que não acurácia:** com 26,5% de churn, um "modelo" que prevê que ninguém sai acerta 73,5% e não encontra nenhum cliente em risco.

**Métricas técnicas**
| Métrica | Pergunta que responde | Uso |
|---|---|---|
| **AUC-ROC** (principal) | Sorteando um cliente que cancelou e um que ficou, qual a chance de o modelo dar probabilidade maior ao que cancelou? (0,5 = sorteio; 1,0 = perfeito) | Comparar modelos na validação cruzada, sem depender de limiar |
| Recall no top 20% | De todos que iam cancelar, quantos estão na lista? | Reportar o modelo final |
| Precision no top 20% | De quem está na lista, quantos iam mesmo cancelar? | Reportar o modelo final |
| Precision, recall, F1 e matriz de confusão (classe churn) | Desempenho da classificação 0/1 no limiar escolhido | Reportar o modelo final |

**Métrica de negócio: % da receita em risco capturada no top 20%**

```
receita capturada = Σ Monthly Charges dos churners no top 20%
                    ─────────────────────────────────────────────
                    Σ Monthly Charges de todos os churners
```

- **Referência:** uma lista aleatória de 20% captura ~20% da receita em risco.
- **Leitura:** se o modelo capturar 50%, abordar 1 em cada 5 clientes alcança metade da receita que seria perdida, 2,5× mais que escolher ao acaso.
- **Ligação com a métrica técnica:** quanto melhor a ordenação (AUC), mais churners, e mais receita, entram no topo da lista.

**Protocolo:** separação treino/teste estratificada; escolha de modelos e hiperparâmetros por validação cruzada (`StratifiedKFold`); o conjunto de teste é avaliado uma única vez, no modelo campeão.

## 8. Predição e implantação
- Pipeline completo (pré-processamento + modelo) salvo em `models/` com `joblib`.
- API FastAPI: `/health` e `/predict` (devolve probabilidade e classe).
- Lote por ciclo para gerar a lista da campanha.

## 9. Monitoramento e riscos
- **Monitorar:** distribuição das features (drift), taxa de churn real, AUC e receita capturada a cada ciclo, conforme os rótulos chegam.
- **Retreino:** quando as métricas caírem de forma consistente ou o perfil da base mudar (novos planos, preços).
- **LGPD:** dados pessoais usados só para retenção; sem exportar identificadores na saída analítica; acesso restrito.
- **Fairness:** comparar o desempenho por `Gender` e `Senior Citizen` para evitar tratamento desigual.
- **Limitações:** dataset de uma única região e de um único trimestre; a ação de retenção pode não representar bem o contexto geral.

# projeto1-telco-churn

Tech Challenge da Fase 1 - Pipeline preditivo de churn para uma operadora de telecomunicações, da análise exploratória ao modelo servido via API.

> 🚧 Projeto em desenvolvimento.

## Contexto
Uma operadora está perdendo clientes em ritmo acelerado. O objetivo é classificar clientes por risco de cancelamento para orientar ações de retenção.

## Dataset
IBM Telco Customer Churn, versão estendida: 7.043 clientes e 33 variáveis (Califórnia, EUA).
Arquivo em `data/raw/Telco_customer_churn.xlsx`; dicionário de dados em `data/raw/data_description.txt`.

## Estrutura
```
data/raw/        dados brutos (versionados)
data/processed/  dados gerados (não versionados)
notebooks/       experimentação: EDA, baseline, modelagem
src/telco_churn/ código produtivo
models/          modelo final serializado
tests/           testes automatizados (pytest)
docs/            ML Canvas, experimentos, Model Card
```

## Como executar
Pré-requisito: [uv](https://docs.astral.sh/uv/) instalado.

```bash
git clone https://github.com/lsapage/projeto1-telco-churn.git
cd projeto1-telco-churn
uv sync
```

## Status
- [x] Estrutura do repositório
- [ ] ML Canvas e métricas
- [ ] EDA e baseline (Regressão Logística)
- [ ] Modelagem (Random Forest, MLP) e comparação
- [ ] Refatoração, testes e API (FastAPI)
- [ ] Model Card e documentação final
"""Configurações centrais do projeto: caminhos, semente e papel de cada coluna."""

from pathlib import Path

# config.py está em src/telco_churn/ → parents[2] é a raiz do repositório
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Caminhos
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Reprodutibilidade
SEED = 42

# Capacidade de abordagem: fração dos clientes de maior risco na lista de retenção
TOP_K = 0.2

# Alvo
TARGET = "Churn Value"  # 1 = cliente cancelou

# Colunas fora das features
LEAKAGE_COLS = ["Churn Label", "Churn Reason", "Churn Score"]
NO_INFO_COLS = ["CustomerID", "Count", "Country", "State"]
GEO_COLS = ["City", "Zip Code", "Lat Long", "Latitude", "Longitude"]
OUT_OF_SCOPE_COLS = ["CLTV"]
SENSITIVE_COLS = ["Gender"]  # fora das features, mas usada na auditoria de fairness

DROP_COLS = LEAKAGE_COLS + NO_INFO_COLS + GEO_COLS + OUT_OF_SCOPE_COLS + SENSITIVE_COLS

# Features que entram no modelo, separadas pelo tratamento que recebem
NUMERIC_COLS = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL_COLS = [
    # Perfil do cliente
    "Senior Citizen",
    "Partner",
    "Dependents",
    # Serviços contratados
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    # Contrato e cobrança
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

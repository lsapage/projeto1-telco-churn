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

# Alvo
TARGET = "Churn Value"  # 1 = cliente cancelou

# Colunas fora das features
LEAKAGE_COLS = ["Churn Label", "Churn Reason", "Churn Score"]
NO_INFO_COLS = ["CustomerID", "Count", "Country", "State", "Lat Long"]
OUT_OF_SCOPE_COLS = ["CLTV"]

DROP_COLS = LEAKAGE_COLS + NO_INFO_COLS + OUT_OF_SCOPE_COLS
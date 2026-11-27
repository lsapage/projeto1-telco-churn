"""Carga e limpeza do dataset Telco Customer Churn."""

from pathlib import Path

import pandas as pd

from telco_churn.config import DATA_RAW_PATH, DROP_COLS, TARGET

# Quem não tem o serviço base também não tem o serviço adicional
NO_SERVICE = {"No internet service": "No", "No phone service": "No"}


def load_raw(path: Path = DATA_RAW_PATH) -> pd.DataFrame:
    """Lê o xlsx original sem nenhuma transformação."""
    return pd.read_excel(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige tipos e aplica as regras de negócio. Não altera o DataFrame recebido."""
    df = df.copy()

    # Total cobrado vem como texto; cliente sem meses de contrato ainda não foi faturado
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df.loc[df["Tenure Months"] == 0, "Total Charges"] = 0

    df = df.replace(NO_SERVICE)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as features (X) do alvo (y)."""
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]
    return X, y
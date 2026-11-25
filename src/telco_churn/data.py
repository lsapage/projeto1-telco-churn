"""Carga e limpeza do dataset Telco Customer Churn."""

from pathlib import Path
import pandas as pd
from telco_churn.config import DATA_RAW_PATH


def load_raw(path: Path = DATA_RAW_PATH) -> pd.DataFrame:
    """Lê o xlsx original sem nenhuma transformação."""
    return pd.read_excel(path)
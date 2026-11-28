"""Fixtures compartilhadas pelos testes."""

import pandas as pd
import pytest

from telco_churn.data import clean, load_raw


@pytest.fixture
def df_sintetico() -> pd.DataFrame:
    """Poucos clientes montados à mão, um para cada regra do clean()."""
    return pd.DataFrame(
        {
            # novo (ainda não faturado), cliente normal, cliente com total inválido
            "Tenure Months": [0, 5, 3],
            "Total Charges": [" ", "250.5", " "],
            "Internet Service": ["DSL", "No", "Fiber optic"],
            "Online Security": ["Yes", "No internet service", "No"],
            "Phone Service": ["Yes", "Yes", "No"],
            "Multiple Lines": ["No", "Yes", "No phone service"],
        }
    )


@pytest.fixture(scope="session")
def df_real() -> pd.DataFrame:
    """Dataset completo já limpo; lido uma única vez por execução."""
    return clean(load_raw())
"""Testes do pré-processador (build_preprocessor)."""

import numpy as np
import pandas as pd
import pytest

from telco_churn.data import split_features_target
from telco_churn.features import build_preprocessor


@pytest.fixture(scope="module")
def X_real(df_real) -> pd.DataFrame:
    X, _ = split_features_target(df_real)
    return X


@pytest.fixture(scope="module")
def prep(X_real):
    """Pré-processador já ajustado no dado real, com saída em DataFrame."""
    return build_preprocessor().set_output(transform="pandas").fit(X_real)


def test_saida_numerica_completa(prep, X_real):
    saida = prep.transform(X_real)

    # 3 numéricas + 12 binárias + Internet (3) + Contract (3) + Payment (4)
    assert saida.shape == (len(X_real), 25)
    assert not saida.isna().any().any()


def test_cliente_sozinho_usa_valores_do_treino(prep, X_real):
    # Na produção chega um cliente por vez: o resultado não pode depender do lote
    lote = prep.transform(X_real)
    sozinho = prep.transform(X_real.iloc[[0]])

    pd.testing.assert_frame_equal(sozinho, lote.iloc[[0]])


def test_categoria_desconhecida_vira_zeros(prep, X_real):
    cliente = X_real.iloc[[0]].copy()
    cliente["Contract"] = "Quinquennial"

    with pytest.warns(UserWarning, match="unknown categories"):
        saida = prep.transform(cliente)

    contrato = saida.filter(like="Contract_")
    assert (contrato == 0).all().all()


def test_valor_faltante_vira_mediana_do_treino(prep, X_real):
    faltante = X_real.iloc[[0]].copy()
    faltante["Total Charges"] = np.nan
    mediana = X_real.iloc[[0]].copy()
    mediana["Total Charges"] = X_real["Total Charges"].median()

    assert prep.transform(faltante).equals(prep.transform(mediana))

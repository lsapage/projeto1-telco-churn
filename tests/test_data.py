"""Testes da limpeza (clean) e da separação X/y (split_features_target)."""

import pandas as pd

from telco_churn.config import DROP_COLS, TARGET
from telco_churn.data import NO_SERVICE, clean, split_features_target

# clean()


def test_total_charges_vira_numerico(df_sintetico):
    resultado = clean(df_sintetico)

    assert pd.api.types.is_numeric_dtype(resultado["Total Charges"])


def test_cliente_sem_meses_tem_total_zero(df_sintetico):
    resultado = clean(df_sintetico)

    novos = resultado["Tenure Months"] == 0
    assert (resultado.loc[novos, "Total Charges"] == 0).all()


def test_total_invalido_com_meses_fica_nulo(df_sintetico):
    # Não inventamos valor aqui: o imputador do pipeline decide
    resultado = clean(df_sintetico)

    assert pd.isna(resultado.loc[2, "Total Charges"])


def test_remove_terceira_opcao_dos_servicos(df_sintetico):
    resultado = clean(df_sintetico)

    assert not resultado.isin(list(NO_SERVICE)).any().any()


def test_nao_altera_dataframe_recebido(df_sintetico):
    original = df_sintetico.copy()

    clean(df_sintetico)

    pd.testing.assert_frame_equal(df_sintetico, original)


# split_features_target()


def test_features_sem_colunas_excluidas_nem_alvo(df_real):
    X, _ = split_features_target(df_real)

    proibidas = set(DROP_COLS) | {TARGET}
    vazadas = proibidas & set(X.columns)
    assert vazadas == set()


def test_genero_fora_das_features(df_real):
    X, _ = split_features_target(df_real)

    assert "Gender" not in X.columns


def test_formato_com_dado_real(df_real):
    X, y = split_features_target(df_real)

    assert len(X) == len(y) == len(df_real)  # nenhuma linha perdida
    assert X.shape[1] == 18  # contrato com o pipeline
    assert set(y.unique()) == {0, 1}
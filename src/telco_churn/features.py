"""Pré-processamento: transforma as features limpas na matriz numérica que o modelo recebe."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_churn.config import CATEGORICAL_COLS, NUMERIC_COLS


def build_preprocessor() -> ColumnTransformer:
    """Cria o pré-processador ainda não ajustado.

    Ele aprende medianas, categorias e escalas no fit; por isso fica dentro do
    pipeline do modelo e é ajustado só com dados de treino.
    """
    # Valores: preenche faltantes com a mediana e põe tudo na mesma escala
    numericas = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    # Categorias: preenche com a mais comum; Yes/No vira uma coluna 0/1,
    # as demais viram uma coluna por opção; categoria nova não quebra a previsão
    categoricas = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
    )

    return ColumnTransformer(
        [
            ("num", numericas, NUMERIC_COLS),
            ("cat", categoricas, CATEGORICAL_COLS),
        ]
    )

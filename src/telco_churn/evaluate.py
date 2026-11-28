"""Métricas da lista de retenção: os k% clientes com maior probabilidade de churn."""

import math

import numpy as np
import pandas as pd

from telco_churn.config import TOP_K


def top_k_positions(y_score, k=TOP_K):
    """Posições dos clientes que entram na lista de retenção.

    A lista tem ceil(k * n) clientes, os de maior probabilidade de churn.
    Em caso de empate, vale a ordem original dos clientes.
    """
    score = pd.Series(np.asarray(y_score))
    tamanho_lista = math.ceil(k * len(score))
    ranking = score.sort_values(ascending=False, kind="stable")
    return ranking.head(tamanho_lista).index


def recall_at_k(y_true, y_score, k=TOP_K):
    """Dos clientes que cancelaram, que fração está na lista."""
    churn = pd.Series(np.asarray(y_true))
    lista = top_k_positions(y_score, k)
    return churn[lista].sum() / churn.sum()


def precision_at_k(y_true, y_score, k=TOP_K):
    """Dos clientes da lista, que fração de fato cancelou."""
    churn = pd.Series(np.asarray(y_true))
    lista = top_k_positions(y_score, k)
    return churn[lista].mean()


def revenue_captured_at_k(y_true, y_score, revenue, k=TOP_K):
    """Da receita mensal dos clientes que cancelaram, que fração está na lista."""
    churn = pd.Series(np.asarray(y_true))
    receita = pd.Series(np.asarray(revenue, dtype=float))
    receita_em_risco = receita * churn  # quem não cancelou não tem receita em risco
    lista = top_k_positions(y_score, k)
    return receita_em_risco[lista].sum() / receita_em_risco.sum()

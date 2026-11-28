"""Testes das métricas da lista de retenção."""

import numpy as np
import pytest

from telco_churn.evaluate import (
    precision_at_k,
    recall_at_k,
    revenue_captured_at_k,
    top_k_positions,
)

# 10 clientes, 3 churners (posições 0, 1 e 4).
# Com k = 0.2 a lista tem 2 clientes: posições 0 (churner) e 2 (não churner).
Y_TRUE = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
Y_SCORE = [0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.4, 0.05, 0.6, 0.5]
# Receita em risco = receita dos churners = 100 + 50 + 30 = 180
REVENUE = [100, 50, 80, 20, 30, 10, 10, 10, 10, 10]


def test_tamanho_da_lista_arredonda_para_cima():
    assert len(top_k_positions(np.zeros(10), k=0.2)) == 2
    assert len(top_k_positions(np.zeros(7043), k=0.2)) == 1409  # 1408,6 → 1409


def test_lista_tem_as_maiores_probabilidades():
    assert list(top_k_positions(Y_SCORE, k=0.2)) == [0, 2]


def test_empate_respeita_ordem_original():
    assert list(top_k_positions([0.5, 0.5, 0.5, 0.5, 0.5], k=0.4)) == [0, 1]


def test_metricas_no_caso_pequeno():
    assert recall_at_k(Y_TRUE, Y_SCORE, k=0.2) == pytest.approx(1 / 3)
    assert precision_at_k(Y_TRUE, Y_SCORE, k=0.2) == pytest.approx(1 / 2)
    receita = revenue_captured_at_k(Y_TRUE, Y_SCORE, REVENUE, k=0.2)
    assert receita == pytest.approx(100 / 180)


def test_lista_maior_captura_mais():
    # Com k = 0.3 entra a posição 4 (churner, receita 30)
    assert recall_at_k(Y_TRUE, Y_SCORE, k=0.3) == pytest.approx(2 / 3)
    receita = revenue_captured_at_k(Y_TRUE, Y_SCORE, REVENUE, k=0.3)
    assert receita == pytest.approx(130 / 180)


def test_modelo_perfeito():
    # Lista de 2 clientes com 3 churners: só cabem churners
    assert precision_at_k(Y_TRUE, Y_TRUE, k=0.2) == 1.0
    # Lista de 3 clientes: cabem todos os churners
    assert recall_at_k(Y_TRUE, Y_TRUE, k=0.3) == 1.0


def test_modelo_invertido_nao_captura_ninguem():
    score_invertido = [-s for s in Y_TRUE]
    assert recall_at_k(Y_TRUE, score_invertido, k=0.2) == 0.0


def test_lista_aleatoria_captura_cerca_de_k():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=10_000)
    score_aleatorio = rng.random(10_000)
    assert recall_at_k(y, score_aleatorio, k=0.2) == pytest.approx(0.2, abs=0.02)

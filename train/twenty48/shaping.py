"""Shaping posicional do alvo de valor: recompensa peças grandes nos CANTOS,
pune no CENTRO, recompensa pouco nas BORDAS — proporcional ao valor da peça.

pos(board) = (w_corner·Σcanto v + w_edge·Σborda v + w_center·Σcentro v) / Σv,
  v = 2^exp (0 se vazia).  ∈ ~[w_center, w_corner] (tipicamente [-1, +1]).

Classificação de célula em n×n: canto = ambos os eixos na borda; borda = um eixo
na borda; centro = interior (nenhum eixo na borda). Em 4×4 dá 4 cantos / 8 bordas
/ 4 centro, como especificado.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def cell_weights(n: int, w_corner: float, w_edge: float, w_center: float) -> np.ndarray:
    w = np.empty(n * n, dtype=np.float64)
    for r in range(n):
        for c in range(n):
            on_r = r in (0, n - 1)
            on_c = c in (0, n - 1)
            if on_r and on_c:
                w[r * n + c] = w_corner
            elif on_r or on_c:
                w[r * n + c] = w_edge
            else:
                w[r * n + c] = w_center
    return w


def positional_bonus(states, w: np.ndarray) -> np.ndarray:
    """Bônus posicional por estado (mesmo tamanho no lote). Retorna (B,) float64."""
    exps = np.array([s.cells for s in states], dtype=np.int64)  # (B, n*n) expoentes
    vals = np.where(exps > 0, np.left_shift(1, exps), 0).astype(np.float64)  # 2^exp
    tot = vals.sum(1)
    num = vals @ w
    return np.divide(num, tot, out=np.zeros_like(num), where=tot > 0)

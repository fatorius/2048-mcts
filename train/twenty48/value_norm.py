"""Normalização adaptativa do alvo de valor.

Problema (visto no primeiro run): os resultados de partidas se concentram numa
faixa estreita; com `normalize_score = log2/16` isso vira ~0.05 de spread, então
o value head só prevê a média (perda ~0) e o MCTS perde o sinal de valor.

Correção: padronizar o score por tamanho — z = (score - μ)/σ com μ,σ por média
móvel exponencial (acompanha a melhora do jogo) — e passar por sigmoide → alvo
bem espalhado em [0,1], adaptativo, e ainda dentro do contrato de valor [0,1].
Por-tamanho porque scores de 6×6 >> 4×4.
"""

from __future__ import annotations

import math

import numpy as np


class ValueNormalizer:
    def __init__(self, momentum: float = 0.02, min_std: float = 1.0):
        self.momentum = momentum
        self.min_std = min_std
        self._mean: dict[int, float] = {}
        self._sq: dict[int, float] = {}  # EMA de x^2
        self._count: dict[int, int] = {}

    def update(self, size: int, score: float) -> None:
        x = float(score)
        a = self.momentum
        if size not in self._mean:
            self._mean[size] = x
            self._sq[size] = x * x
            self._count[size] = 1
        else:
            self._mean[size] = (1 - a) * self._mean[size] + a * x
            self._sq[size] = (1 - a) * self._sq[size] + a * x * x
            self._count[size] += 1

    def _mu_sigma(self, size: int) -> tuple[float, float]:
        if size not in self._mean:
            return 0.0, 0.0
        mu = self._mean[size]
        var = max(self._sq[size] - mu * mu, 0.0)
        return mu, math.sqrt(var)

    def normalize(self, score: float, size: int) -> float:
        """score bruto → [0,1] padronizado. 0.5 enquanto não há spread suficiente."""
        mu, sigma = self._mu_sigma(size)
        if sigma < self.min_std:
            return 0.5
        z = (float(score) - mu) / sigma
        return 1.0 / (1.0 + math.exp(-z))

    def normalize_array(self, scores, size: int) -> np.ndarray:
        mu, sigma = self._mu_sigma(size)
        if sigma < self.min_std:
            return np.full(len(scores), 0.5, dtype=np.float32)
        z = (np.asarray(scores, dtype=np.float64) - mu) / sigma
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def terminal_value_fn(self):
        """Fn para o MCTS avaliar folhas terminais na MESMA escala do alvo."""
        return lambda state: self.normalize(state.score, state.size)

    def state_dict(self) -> dict:
        return {"mean": dict(self._mean), "sq": dict(self._sq), "count": dict(self._count)}

    def load_state_dict(self, state: dict) -> None:
        # As chaves de tamanho podem virar str após serialização; normaliza p/ int.
        self._mean = {int(k): v for k, v in state.get("mean", {}).items()}
        self._sq = {int(k): v for k, v in state.get("sq", {}).items()}
        self._count = {int(k): v for k, v in state.get("count", {}).items()}

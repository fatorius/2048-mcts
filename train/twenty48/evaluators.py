"""Avaliadores em lote para o MCTS — a fronteira `evaluate` (mesma ideia do TS).

Interface: `evaluator(states) -> (policies (B,4) float32, values (B,) float32)`.
  - uniform_evaluator : prior uniforme + 0.5 (testes/paridade).
  - RolloutEvaluator  : rollout aleatório (warm start da iteração 0).
  - NetEvaluator      : forward em lote da rede (self-play guiado pela rede).
"""

from __future__ import annotations

import numpy as np

from .board import (
    GameState,
    apply_move,
    is_terminal,
    legal_actions,
    normalize_score,
    spawn_random,
)


def uniform_evaluator(states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
    b = len(states)
    return np.full((b, 4), 0.25, dtype=np.float32), np.full(b, 0.5, dtype=np.float32)


class RolloutEvaluator:
    """Valor = média de rollouts aleatórios até o fim (normalizado). Política
    uniforme. Usado só na iteração 0 (warm start) — reaproveita a força da busca
    da Fase 1 em vez do value head aleatório de uma rede recém-inicializada."""

    def __init__(self, rng: np.random.Generator, rollouts: int = 1):
        self.rng = rng
        self.rollouts = max(1, rollouts)

    def _rollout(self, state: GameState) -> float:
        cells, score, size = state.cells, state.score, state.size
        while True:
            cur = GameState(size, cells, score)
            if is_terminal(cur):
                return score
            legal = legal_actions(cur)
            if not legal:
                return score
            action = legal[int(self.rng.integers(len(legal)))]
            cells, gained, _ = apply_move(cur, action)
            score += gained
            cells = spawn_random(cells, self.rng)

    def __call__(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        pols = np.full((len(states), 4), 0.25, dtype=np.float32)
        vals = np.empty(len(states), dtype=np.float32)
        for i, s in enumerate(states):
            if is_terminal(s):
                vals[i] = normalize_score(s.score)
                continue
            total = sum(self._rollout(s) for _ in range(self.rollouts))
            vals[i] = normalize_score(total / self.rollouts)
        return pols, vals


class NetEvaluator:
    """Forward em lote da rede. Assume estados de MESMO tamanho no lote (garantido
    dentro de uma busca)."""

    def __init__(self, net, device: str):
        import torch  # import tardio: RolloutEvaluator não depende de torch

        self._torch = torch
        self.net = net
        self.device = device

    def __call__(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        from .encode import encode_batch

        torch = self._torch
        x = torch.from_numpy(encode_batch(states)).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
            pols = torch.softmax(logits, dim=1).float().cpu().numpy()
            vals = value.float().cpu().numpy()
        return pols.astype(np.float32), vals.astype(np.float32)

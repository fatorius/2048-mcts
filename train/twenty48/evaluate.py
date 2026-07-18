"""Métricas de progresso — joga partidas gulosas (temp 0, sem noise) com a rede.

Em 2048 não há oponente: o sinal de progresso é direto (score médio, taxa de
2048/4096). É o que dizemos ter "melhorado" entre iterações.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .board import initial_state, is_terminal, max_exponent, step
from .evaluators import NetEvaluator
from .mcts import MctsConfig, run_mcts, select_move


@dataclass
class EvalMetrics:
    games: int
    mean_score: float
    best_tile: int
    reach_2048_rate: float
    reach_4096_rate: float
    tile_hist: dict[int, int] = field(default_factory=dict)


def evaluate_net(
    net,
    device: str,
    rng: np.random.Generator,
    size: int = 4,
    games: int = 10,
    sims: int = 100,
    c_puct: float = 1.5,
    batch_size: int = 32,
    move_cap: int = 4000,
) -> EvalMetrics:
    net.eval()
    evaluator = NetEvaluator(net, device)
    cfg = MctsConfig(simulations=sims, c_puct=c_puct, batch_size=batch_size)

    scores: list[int] = []
    exps: list[int] = []
    for _ in range(games):
        state = initial_state(size, rng)
        moves = 0
        while not is_terminal(state) and moves < move_cap:
            result, _ = run_mcts(state, evaluator, rng, cfg, add_noise=False)
            if result.best_action == -1:
                break
            action = select_move(result.visits, 0.0, rng)  # guloso
            state, _ = step(state, action, rng)
            moves += 1
        scores.append(state.score)
        exps.append(max_exponent(state))

    hist: dict[int, int] = {}
    for e in exps:
        tile = 1 << e
        hist[tile] = hist.get(tile, 0) + 1
    return EvalMetrics(
        games=games,
        mean_score=float(np.mean(scores)),
        best_tile=1 << max(exps),
        reach_2048_rate=float(np.mean([e >= 11 for e in exps])),
        reach_4096_rate=float(np.mean([e >= 12 for e in exps])),
        tile_hist=hist,
    )

"""Self-play — gera uma partida com MCTS e registra as posições no buffer.

Por posição grava (board, distribuição de visitas normalizada = alvo de política).
Ao fim, o alvo de valor de TODAS as posições é o resultado normalizado da partida
(z = normalize_score(score final)) — o alvo de outcome padrão do AlphaZero.

Dirichlet noise na raiz + temperatura (explora no começo, explota no fim).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board import initial_state, is_terminal, max_exponent, normalize_score, step
from .buffer import ReplayBuffer
from .mcts import MctsConfig, run_mcts, select_move


@dataclass
class GameStats:
    size: int
    score: int
    max_exponent: int
    moves: int


def self_play_game(
    evaluator,
    rng: np.random.Generator,
    mcts_cfg: MctsConfig,
    size: int,
    buffer: ReplayBuffer,
    temp_moves: int = 20,
    move_cap: int = 4000,
) -> GameStats:
    state = initial_state(size, rng)
    pending: list[tuple] = []  # (state, policy_target)
    moves = 0

    while not is_terminal(state) and moves < move_cap:
        result, _ = run_mcts(state, evaluator, rng, mcts_cfg, add_noise=True)
        if result.best_action == -1:
            break

        visits = np.asarray(result.visits, dtype=np.float64)
        total = visits.sum()
        policy = (visits / total).astype(np.float32) if total > 0 else np.full(4, 0.25, np.float32)
        pending.append((state, policy))

        temperature = 1.0 if moves < temp_moves else 0.0
        action = select_move(result.visits, temperature, rng)
        if action == -1:
            break
        state, _ = step(state, action, rng)
        moves += 1

    z = normalize_score(state.score)
    for st, pol in pending:
        buffer.add(size, st, pol, z)

    return GameStats(size=size, score=state.score, max_exponent=max_exponent(state), moves=moves)

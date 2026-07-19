"""Self-play e eval por PARTIDAS PARALELAS.

Cada partida roda uma busca SEQUENCIAL (corrotina `mcts_search_gen` — uma folha
por simulação, sem virtual loss, = mesma busca do TS). O driver avança todas as
buscas ativas em lockstep e junta as folhas pendentes de partidas DIFERENTES num
único forward de GPU. Assim recupera-se a qualidade da busca sequencial (que o
batching por folhas de uma mesma busca destruía) SEM perder throughput de GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .board import GameState, initial_state, is_terminal, max_exponent, step
from .buffer import ReplayBuffer
from .evaluate import EvalMetrics
from .mcts import MctsConfig, mcts_search_gen, select_move
from .self_play import GameStats
from .value_norm import ValueNormalizer


@dataclass
class _Slot:
    index: int
    size: int
    rng: np.random.Generator
    state: GameState
    gen: object = None  # busca ativa (gerador) ou None entre lances
    pending: GameState | None = None  # folha aguardando avaliação
    records: list = field(default_factory=list)  # (state, policy) — só self-play
    moves: int = 0
    done: bool = False


def _run_parallel(slots, evaluator, cfg, tvf, add_noise, temp_moves, move_cap, record, on_finish):
    def _finish(slot: _Slot):
        slot.done = True
        slot.gen = None
        slot.pending = None
        on_finish(slot)

    def _apply_move(slot: _Slot, result):
        if result.best_action == -1:
            _finish(slot)
            return
        if record:
            visits = np.asarray(result.visits, dtype=np.float64)
            total = visits.sum()
            policy = (
                (visits / total).astype(np.float32) if total > 0 else np.full(4, 0.25, np.float32)
            )
            slot.records.append((slot.state, policy))
        temperature = 1.0 if slot.moves < temp_moves else 0.0
        action = select_move(result.visits, temperature, slot.rng)
        if action == -1:
            _finish(slot)
            return
        slot.state, _ = step(slot.state, action, slot.rng)
        slot.moves += 1  # gen fica None -> próximo lance inicia na próxima rodada

    def _advance(slot: _Slot, sent):
        try:
            slot.pending = slot.gen.send(sent) if sent is not None else next(slot.gen)
        except StopIteration as e:
            slot.gen = None
            slot.pending = None
            _apply_move(slot, e.value)

    while True:
        # Toda partida viva sem busca ativa inicia a busca do lance atual (ou termina).
        for slot in slots:
            if slot.done or slot.gen is not None:
                continue
            if is_terminal(slot.state) or slot.moves >= move_cap:
                _finish(slot)
                continue
            slot.gen = mcts_search_gen(slot.state, cfg, slot.rng, add_noise, tvf)
            _advance(slot, None)  # prime: primeiro yield (raiz)

        batch = [s for s in slots if s.gen is not None and s.pending is not None]
        if not batch:
            break
        policies, values = evaluator([s.pending for s in batch])
        for k, slot in enumerate(batch):
            _advance(slot, (policies[k], values[k]))


def _make_slots(n, sizes, rng):
    slots = []
    for i in range(n):
        size = int(rng.choice(sizes))
        grng = np.random.default_rng(int(rng.integers(1 << 62)))
        slots.append(_Slot(index=i, size=size, rng=grng, state=initial_state(size, grng)))
    return slots


def play_games_parallel(
    n_games: int,
    sizes,
    evaluator,
    rng: np.random.Generator,
    cfg: MctsConfig,
    buffer: ReplayBuffer,
    normalizer: ValueNormalizer,
    temp_moves: int = 20,
    move_cap: int = 4000,
    on_game=None,
) -> list[GameStats]:
    """Self-play de `n_games` partidas em paralelo. Grava posições no buffer e
    atualiza o normalizador. Retorna GameStats por partida (na ordem dos slots)."""
    tvf = normalizer.terminal_value_fn()
    slots = _make_slots(n_games, sizes, rng)
    stats: list[GameStats | None] = [None] * n_games

    def on_finish(slot: _Slot):
        raw = float(slot.state.score)
        for st, pol in slot.records:
            buffer.add(slot.size, st, pol, raw)
        normalizer.update(slot.size, slot.state.score)
        stats[slot.index] = GameStats(
            slot.size, slot.state.score, max_exponent(slot.state), slot.moves
        )
        if on_game is not None:
            on_game(slot.index, slot.state.score, max_exponent(slot.state), slot.moves)

    _run_parallel(slots, evaluator, cfg, tvf, True, temp_moves, move_cap, record=True, on_finish=on_finish)
    return [s for s in stats if s is not None]


def evaluate_parallel(
    evaluator,
    rng: np.random.Generator,
    size: int = 4,
    games: int = 10,
    sims: int = 100,
    c_puct: float = 1.5,
    move_cap: int = 4000,
    terminal_value_fn=None,
    on_game=None,
) -> EvalMetrics:
    """Partidas gulosas (temp 0, sem noise) em paralelo — rede ou rollout."""
    cfg = MctsConfig(simulations=sims, c_puct=c_puct)
    slots = _make_slots(games, (size,), rng)
    out: list[tuple[int, int]] = [(0, 0)] * games

    def on_finish(slot: _Slot):
        exp = max_exponent(slot.state)
        out[slot.index] = (slot.state.score, exp)
        if on_game is not None:
            on_game(slot.index, slot.state.score, exp, slot.moves)

    _run_parallel(slots, evaluator, cfg, terminal_value_fn, False, 0, move_cap, record=False, on_finish=on_finish)

    scores = [s for s, _ in out]
    exps = [e for _, e in out]
    hist: dict[int, int] = {}
    for e in exps:
        hist[1 << e] = hist.get(1 << e, 0) + 1
    return EvalMetrics(
        games=games,
        mean_score=float(np.mean(scores)),
        best_tile=1 << max(exps),
        reach_2048_rate=float(np.mean([e >= 11 for e in exps])),
        reach_4096_rate=float(np.mean([e >= 12 for e in exps])),
        tile_hist=hist,
    )

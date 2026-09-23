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


def _run_parallel(slots, evaluator, cfg, tvf, add_noise, temp_moves, move_cap, record, on_finish, transform_fn=None):
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
            vt = transform_fn(slot.size) if transform_fn is not None else None
            slot.gen = mcts_search_gen(slot.state, cfg, slot.rng, add_noise, tvf, vt)
            _advance(slot, None)  # prime: primeiro yield (raiz)

        batch = [s for s in slots if s.gen is not None and s.pending is not None]
        if not batch:
            break
        policies, values = evaluator([s.pending for s in batch])
        for k, slot in enumerate(batch):
            _advance(slot, (policies[k], values[k]))


def load_start_pool(path) -> dict[int, list[GameState]]:
    """Carrega um pool de posições iniciais (JSONL {size,cells,score}) agrupado por
    tamanho. Usado p/ self-play de ENDGAME: começa as partidas de posições de
    meio/fim de jogo em vez do tabuleiro vazio (foca o treino na fronteira)."""
    import json

    pool: dict[int, list[GameState]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pool.setdefault(int(r["size"]), []).append(
                GameState(int(r["size"]), tuple(r["cells"]), int(r["score"]))
            )
    return pool


def _make_slots(n, sizes, rng, start_pool=None):
    # Tamanho escolhido UNIFORMEMENTE entre `sizes` (equilíbrio por tamanho
    # independe do pool ser desbalanceado); a posição inicial vem do pool daquele
    # tamanho quando há, senão tabuleiro vazio.
    slots = []
    for i in range(n):
        size = int(rng.choice(sizes))
        grng = np.random.default_rng(int(rng.integers(1 << 62)))
        if start_pool and start_pool.get(size):
            bucket = start_pool[size]
            state = bucket[int(grng.integers(len(bucket)))]
        else:
            state = initial_state(size, grng)
        slots.append(_Slot(index=i, size=size, rng=grng, state=state))
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
    start_pool=None,
) -> list[GameStats]:
    """Self-play de `n_games` partidas em paralelo. Grava posições no buffer e
    atualiza o normalizador. Retorna GameStats por partida (na ordem dos slots).
    Com `start_pool`, as partidas começam de posições de meio/fim de jogo."""
    tvf = normalizer.terminal_value_fn()
    slots = _make_slots(n_games, sizes, rng, start_pool)
    stats: list[GameStats | None] = [None] * n_games

    def on_finish(slot: _Slot):
        # Alvo = reward-to-go bruto (final − score na posição); atualiza o
        # normalizador com a distribuição de reward-to-go. Ver self_play.py.
        final = float(slot.state.score)
        for st, pol in slot.records:
            rtg = final - float(st.score)
            buffer.add(slot.size, st, pol, rtg)
            normalizer.update(slot.size, rtg)
        stats[slot.index] = GameStats(
            slot.size, slot.state.score, max_exponent(slot.state), slot.moves
        )
        if on_game is not None:
            on_game(slot.index, slot.state.score, max_exponent(slot.state), slot.moves)

    _run_parallel(
        slots, evaluator, cfg, tvf, True, temp_moves, move_cap,
        record=True, on_finish=on_finish, transform_fn=normalizer.transform,
    )
    return [s for s in stats if s is not None]


# ---------------------------------------------------------------------------
# Self-play MULTIPROCESSO (lever #2): N workers (1 core cada) rodam o driver
# sequencial acima em fatias das partidas. Enquanto um worker faz CPU (árvore
# MCTS + board.step), o forward de outro roda na GPU — sobrepondo CPU/GPU e
# enchendo a GPU que ficava ~50% ociosa no driver de 1 processo. Cada worker tem
# seu próprio contexto CUDA (~120 MB aqui) e cópia da rede; o normalizador fica
# CONGELADO durante o self-play da iteração (snapshot do início) e é atualizado
# no processo principal com os reward-to-go coletados.
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _mp_init(arch: dict, device: str):
    import torch

    from .net import Net

    # Morrer junto com o processo pai: se o treino for morto (kill), o kernel envia
    # SIGKILL ao worker. Sem isto, workers viram órfãos (reparented p/ init) e
    # continuam consumindo CPU/GPU — 6 workers em 4 cores etc. (Linux-only.)
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass
    torch.set_num_threads(1)  # 1 core por worker (não oversubscrever os 4 vCPUs)
    _WORKER["net"] = Net(**arch).to(device).eval()
    _WORKER["device"] = device


def _selfplay_collect(n_games, sizes, evaluator, rng, cfg, norm, temp_moves, move_cap, start_pool):
    """Igual ao play_games_parallel, mas RETORNA os registros em vez de mutar um
    buffer/normalizador compartilhado (normalizador congelado; ver acima)."""
    tvf = norm.terminal_value_fn()
    slots = _make_slots(n_games, sizes, rng, start_pool)
    records = []  # (size, state, policy, rtg)
    stats = []

    def on_finish(slot: _Slot):
        final = float(slot.state.score)
        for st, pol in slot.records:
            records.append((slot.size, st, pol, final - float(st.score)))
        stats.append(GameStats(slot.size, slot.state.score, max_exponent(slot.state), slot.moves))

    _run_parallel(
        slots, evaluator, cfg, tvf, True, temp_moves, move_cap,
        record=True, on_finish=on_finish, transform_fn=norm.transform,
    )
    return records, stats


def _mp_run(payload: dict):
    import torch

    from .evaluators import NetEvaluator

    net = _WORKER["net"]
    net.load_state_dict(payload["net_state"])  # pesos da iteração (tensores CPU)
    net.eval()
    norm = ValueNormalizer()
    norm.load_state_dict(payload["norm_state"])
    ev = NetEvaluator(net, _WORKER["device"])
    rng = np.random.default_rng(payload["seed"])
    with torch.no_grad():
        return _selfplay_collect(
            payload["n_games"], payload["sizes"], ev, rng, payload["cfg"], norm,
            payload["temp_moves"], payload["move_cap"], payload["start_pool"],
        )


class ParallelSelfPlay:
    """Pool persistente de workers de self-play (CUDA inicializado 1× por worker;
    reusado entre iterações). Cada iteração envia os pesos atuais e agrega os
    registros no buffer/normalizador do processo principal."""

    def __init__(self, arch: dict, device: str, n_workers: int):
        import torch.multiprocessing as mp

        self.n_workers = n_workers
        ctx = mp.get_context("spawn")  # CUDA exige spawn (não fork)
        self.pool = ctx.Pool(n_workers, initializer=_mp_init, initargs=(arch, device))

    def play(self, n_games, sizes, net, normalizer, cfg, buffer, temp_moves, move_cap,
             start_pool, base_seed) -> list[GameStats]:
        net_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        norm_state = normalizer.state_dict()
        per = [
            n_games // self.n_workers + (1 if i < n_games % self.n_workers else 0)
            for i in range(self.n_workers)
        ]
        payloads = [
            dict(net_state=net_state, norm_state=norm_state, cfg=cfg, n_games=g,
                 sizes=tuple(sizes), temp_moves=temp_moves, move_cap=move_cap,
                 start_pool=start_pool, seed=base_seed + 1 + i)
            for i, g in enumerate(per) if g > 0
        ]
        results = self.pool.map(_mp_run, payloads)
        stats: list[GameStats] = []
        for records, st in results:
            for size, state, pol, rtg in records:
                buffer.add(size, state, pol, rtg)
                normalizer.update(size, rtg)
            stats.extend(st)
        return stats

    def close(self):
        self.pool.close()
        self.pool.join()


def evaluate_parallel(
    evaluator,
    rng: np.random.Generator,
    size: int = 4,
    games: int = 10,
    sims: int = 100,
    c_puct: float = 1.5,
    move_cap: int = 4000,
    terminal_value_fn=None,
    value_transform=None,
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

    tfn = (lambda _s: value_transform) if value_transform is not None else None
    _run_parallel(
        slots, evaluator, cfg, terminal_value_fn, False, 0, move_cap,
        record=False, on_finish=on_finish, transform_fn=tfn,
    )

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

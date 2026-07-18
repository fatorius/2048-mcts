"""MCTS com chance nodes explícitos — espelho de play/src/core/mcts.ts, mas com
avaliação de folhas EM LOTE (crítico para eficiência de GPU).

Batching por ondas com virtual loss: cada onda desce `batch_size` vezes até
folhas distintas (o virtual loss diversifica as descidas e os chance nodes
injetam variação por si), avalia todas as folhas num único forward, e então
retropropaga. A semântica de busca é a mesma do TS (backup ponderado pela
distribuição exata de spawn via amostragem; valor somado sem alternância de
sinal, pois 2048 é solo contra o acaso).

O avaliador é a fronteira `evaluate` (evaluators.py): rollout no warm start,
rede no self-play. O núcleo abaixo não muda entre eles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .board import (
    ACTIONS,
    GameState,
    apply_move,
    is_terminal,
    normalize_score,
    spawn_outcomes,
    with_spawn,
)

VLOSS_VALUE = 0.0  # 2048: valor em [0,1]; "perda" virtual = 0 (desencoraja reescolha)


class DecisionNode:
    __slots__ = (
        "state",
        "terminal",
        "expanded",
        "pending",
        "P",
        "N",
        "W",
        "VL",
        "legal",
        "children",
        "total_n",
        "total_vl",
        "eval_value",
    )

    def __init__(self, state: GameState):
        self.state = state
        self.terminal = is_terminal(state)
        self.expanded = False
        self.pending = False  # já enfileirado para avaliação nesta onda
        self.P = [0.0, 0.0, 0.0, 0.0]
        self.N = [0.0, 0.0, 0.0, 0.0]
        self.W = [0.0, 0.0, 0.0, 0.0]
        self.VL = [0, 0, 0, 0]  # virtual loss por aresta
        self.legal = [False, False, False, False]
        self.children: list[ChanceNode | None] = [None, None, None, None]
        self.total_n = 0
        self.total_vl = 0
        self.eval_value = 0.0


class ChanceNode:
    __slots__ = ("cells", "gained", "base_score", "size", "outcomes", "children")

    def __init__(self, cells, gained, base_score, size):
        self.cells = cells
        self.gained = gained
        self.base_score = base_score
        self.size = size
        self.outcomes = spawn_outcomes(cells)
        self.children: dict[int, DecisionNode] = {}


@dataclass
class MctsConfig:
    simulations: int = 128
    c_puct: float = 1.5
    batch_size: int = 32
    dirichlet_alpha: float = 0.5
    dirichlet_eps: float = 0.25


@dataclass
class SearchResult:
    visits: list[float]
    q_values: list[float]
    priors: list[float]
    legal: list[bool]
    best_action: int  # -1 se terminal


def _terminal_value(state: GameState) -> float:
    return normalize_score(state.score)


def _select(node: DecisionNode, c_puct: float) -> int:
    sqrt_total = math.sqrt(node.total_n + node.total_vl)
    best, best_a = -1.0, -1
    for a in ACTIONS:
        if not node.legal[a]:
            continue
        n_eff = node.N[a] + node.VL[a]
        q = node.W[a] / n_eff if n_eff > 0 else 0.0
        u = c_puct * node.P[a] * sqrt_total / (1 + n_eff)
        s = q + u
        if s > best:
            best, best_a = s, a
    return best_a


def _get_chance(node: DecisionNode, a: int) -> ChanceNode:
    existing = node.children[a]
    if existing is not None:
        return existing
    cells, gained, _ = apply_move(node.state, a)
    chance = ChanceNode(cells, gained, node.state.score, node.state.size)
    node.children[a] = chance
    return chance


def _sample_outcome(chance: ChanceNode, rng: np.random.Generator) -> DecisionNode:
    outcomes = chance.outcomes
    r = rng.random()
    chosen = outcomes[-1]
    for o in outcomes:
        r -= o.prob
        if r < 0:
            chosen = o
            break
    key = chosen.index * 4 + chosen.exponent
    cached = chance.children.get(key)
    if cached is not None:
        return cached
    child = DecisionNode(
        GameState(
            size=chance.size,
            cells=with_spawn(chance.cells, chosen.index, chosen.exponent),
            score=chance.base_score + chance.gained,
        )
    )
    chance.children[key] = child
    return child


def _descend(root: DecisionNode, c_puct: float, rng: np.random.Generator):
    """Desce aplicando virtual loss. Retorna (path, leaf|None, terminal_value|None)."""
    path: list[tuple[DecisionNode, int]] = []
    node = root
    while True:
        if node.terminal:
            return path, None, _terminal_value(node.state)
        if not node.expanded:
            return path, node, None
        a = _select(node, c_puct)
        node.VL[a] += 1
        node.total_vl += 1
        path.append((node, a))
        chance = _get_chance(node, a)
        node = _sample_outcome(chance, rng)


def _backup(path: list[tuple[DecisionNode, int]], value: float) -> None:
    for node, a in path:
        node.VL[a] -= 1
        node.total_vl -= 1
        node.N[a] += 1
        node.W[a] += value
        node.total_n += 1


def _expand(nodes: list[DecisionNode], evaluator) -> None:
    """Avalia folhas em lote e as expande (P mascarado/renormalizado por legalidade)."""
    states = [nd.state for nd in nodes]
    policies, values = evaluator(states)
    for i, nd in enumerate(nodes):
        legal_mask = [apply_move(nd.state, a)[2] for a in ACTIONS]
        p = policies[i]
        total = sum(p[a] for a in ACTIONS if legal_mask[a])
        for a in ACTIONS:
            nd.legal[a] = legal_mask[a]
            nd.P[a] = (p[a] / total) if (legal_mask[a] and total > 0) else 0.0
        nd.expanded = True
        nd.eval_value = float(values[i])


def _apply_dirichlet(root: DecisionNode, alpha: float, eps: float, rng: np.random.Generator) -> None:
    legal_idx = [a for a in ACTIONS if root.legal[a]]
    if len(legal_idx) < 2:
        return
    noise = rng.dirichlet([alpha] * len(legal_idx))
    for k, a in enumerate(legal_idx):
        root.P[a] = (1 - eps) * root.P[a] + eps * float(noise[k])


def run_mcts(
    root_state: GameState,
    evaluator,
    rng: np.random.Generator,
    config: MctsConfig,
    add_noise: bool = False,
) -> tuple[SearchResult, DecisionNode]:
    root = DecisionNode(root_state)
    if root.terminal:
        return SearchResult([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [False] * 4, -1), root

    _expand([root], evaluator)
    if add_noise:
        _apply_dirichlet(root, config.dirichlet_alpha, config.dirichlet_eps, rng)

    done = 0
    while done < config.simulations:
        wave = min(config.batch_size, config.simulations - done)
        collected: list[tuple[list, DecisionNode | None, float | None]] = []
        leaves: list[DecisionNode] = []
        for _ in range(wave):
            path, leaf, tval = _descend(root, config.c_puct, rng)
            if leaf is not None and not leaf.pending:
                leaf.pending = True
                leaves.append(leaf)
            collected.append((path, leaf, tval))
            done += 1

        if leaves:
            _expand(leaves, evaluator)

        for path, leaf, tval in collected:
            value = tval if tval is not None else leaf.eval_value  # type: ignore[union-attr]
            _backup(path, value)
        for leaf in leaves:
            leaf.pending = False

    visits = list(root.N)
    q_values = [(root.W[a] / root.N[a]) if root.N[a] > 0 else 0.0 for a in ACTIONS]
    priors = list(root.P)
    best_action, best_visits = -1, -1.0
    for a in ACTIONS:
        if root.legal[a] and root.N[a] > best_visits:
            best_visits, best_action = root.N[a], a
    return SearchResult(visits, q_values, priors, list(root.legal), best_action), root


def select_move(visits: list[float], temperature: float, rng: np.random.Generator) -> int:
    """Escolhe a ação a partir das visitas: τ→0 = argmax, τ=1 = ∝ visitas."""
    v = np.asarray(visits, dtype=np.float64)
    if v.sum() == 0:
        return -1
    if temperature <= 1e-6:
        return int(v.argmax())
    logits = np.log(np.maximum(v, 1e-12)) / temperature
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return int(rng.choice(len(v), p=probs))

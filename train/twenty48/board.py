"""Motor do jogo 2048 em Python — espelho fiel de play/src/core/board.ts.

Mesma semântica exata (expoentes row-major, deslize em direção ao índice 0,
distribuição de spawn exata), para que o MCTS de treino e o MCTS que roda no
browser sejam a MESMA busca. A paridade é verificada contra fixtures geradas
pelo lado TS (ver tests/test_parity.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

# Ações — ordem canônica idêntica ao TS e à cabeça de política da rede.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = (UP, RIGHT, DOWN, LEFT)
ACTION_NAMES = ("Up", "Right", "Down", "Left")

SPAWN_PROB_2 = 0.9
SPAWN_PROB_4 = 0.1

# Normalização de valor — idêntica ao TS (evaluate.ts). Alvo da cabeça de valor.
VALUE_SCALE = 16.0

Cells = tuple[int, ...]


@dataclass(frozen=True)
class GameState:
    size: int
    cells: Cells  # row-major, expoentes (0 = vazio)
    score: int


# ---------------------------------------------------------------------------
# Índices de linha por ação (cacheados por tamanho) — mesma ordem do TS.
# ---------------------------------------------------------------------------

_line_cache: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}


def line_indices(size: int, action: int) -> tuple[tuple[int, ...], ...]:
    key = (size, action)
    cached = _line_cache.get(key)
    if cached is not None:
        return cached
    lines: list[tuple[int, ...]] = []
    for k in range(size):
        line: list[int] = []
        for j in range(size):
            if action == UP:  # coluna k, cima->baixo (cabeça no topo)
                r, c = j, k
            elif action == RIGHT:  # linha k, direita->esquerda
                r, c = k, size - 1 - j
            elif action == DOWN:  # coluna k, baixo->cima
                r, c = size - 1 - j, k
            else:  # LEFT: linha k, esquerda->direita
                r, c = k, j
            line.append(r * size + c)
        lines.append(tuple(line))
    res = tuple(lines)
    _line_cache[key] = res
    return res


# ---------------------------------------------------------------------------
# Deslize + fusão
# ---------------------------------------------------------------------------


def _slide_line(vals: list[int]) -> tuple[list[int], int]:
    out: list[int] = []
    gained = 0
    prev = 0  # expoente pendente aguardando possível fusão (0 = nenhum)
    for v in vals:
        if v == 0:
            continue
        if prev == v:
            merged = v + 1
            out.append(merged)
            gained += 1 << merged
            prev = 0
        else:
            if prev != 0:
                out.append(prev)
            prev = v
    if prev != 0:
        out.append(prev)
    while len(out) < len(vals):
        out.append(0)
    return out, gained


def apply_move(state: GameState, action: int) -> tuple[Cells, int, bool]:
    """Aplica um movimento SEM spawnar. Puro. Retorna (cells, gained, moved)."""
    size = state.size
    cells = list(state.cells)
    gained = 0
    moved = False
    for line in line_indices(size, action):
        vals = [cells[i] for i in line]
        slid, g = _slide_line(vals)
        gained += g
        for j, idx in enumerate(line):
            if cells[idx] != slid[j]:
                cells[idx] = slid[j]
                moved = True
    return tuple(cells), gained, moved


# ---------------------------------------------------------------------------
# Consultas de estado
# ---------------------------------------------------------------------------


def empty_cells(state: GameState) -> list[int]:
    return [i for i, v in enumerate(state.cells) if v == 0]


def legal_actions(state: GameState) -> list[int]:
    return [a for a in ACTIONS if apply_move(state, a)[2]]


def is_terminal(state: GameState) -> bool:
    size, cells = state.size, state.cells
    if 0 in cells:
        return False
    for r in range(size):
        for c in range(size):
            v = cells[r * size + c]
            if c + 1 < size and cells[r * size + c + 1] == v:
                return False
            if r + 1 < size and cells[(r + 1) * size + c] == v:
                return False
    return True


def max_exponent(state: GameState) -> int:
    return max(state.cells) if state.cells else 0


def normalize_score(score: float) -> float:
    v = log2(score + 1) / VALUE_SCALE
    return 0.0 if v < 0 else 1.0 if v > 1 else v


# ---------------------------------------------------------------------------
# Spawn: distribuição exata e amostragem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpawnOutcome:
    index: int
    exponent: int  # 1 ("2") ou 2 ("4")
    prob: float


def spawn_outcomes(cells: Cells) -> list[SpawnOutcome]:
    empties = [i for i, v in enumerate(cells) if v == 0]
    e = len(empties)
    out: list[SpawnOutcome] = []
    for index in empties:
        out.append(SpawnOutcome(index, 1, SPAWN_PROB_2 / e))
        out.append(SpawnOutcome(index, 2, SPAWN_PROB_4 / e))
    return out


def with_spawn(cells: Cells, index: int, exponent: int) -> Cells:
    lst = list(cells)
    lst[index] = exponent
    return tuple(lst)


def empty_state(size: int) -> GameState:
    return GameState(size=size, cells=tuple([0] * size * size), score=0)


# ---------------------------------------------------------------------------
# Amostragem e passo (usam numpy.Generator; a aleatoriedade não precisa casar
# com o mulberry32 do TS — só as operações determinísticas do motor precisam).
# ---------------------------------------------------------------------------


def spawn_random(cells: Cells, rng) -> Cells:
    empties = [i for i, v in enumerate(cells) if v == 0]
    if not empties:
        return cells
    index = empties[int(rng.integers(len(empties)))]
    exponent = 1 if rng.random() < SPAWN_PROB_2 else 2
    return with_spawn(cells, index, exponent)


def initial_state(size: int, rng) -> GameState:
    cells: Cells = tuple([0] * size * size)
    cells = spawn_random(cells, rng)
    cells = spawn_random(cells, rng)
    return GameState(size=size, cells=cells, score=0)


def step(state: GameState, action: int, rng) -> tuple[GameState, bool]:
    """Passo real: aplica a ação e, se legal, spawna. Retorna (novo_estado, moved)."""
    cells, gained, moved = apply_move(state, action)
    if not moved:
        return state, False
    return GameState(state.size, spawn_random(cells, rng), state.score + gained), True

"""Codificação do board no tensor de entrada 20×n×n do plano.

20 canais fixos, independentes de n (preserva o agnosticismo de tamanho):
  - 0..16 : one-hot por expoente e∈[1..17] (canal e-1). Célula vazia = tudo 0.
            O one-hot serve à igualdade/fusão de forma nativa.
  - 17    : vazio (1 onde a célula está vazia) — densidade de vazios = perigo.
  - 18    : log2 normalizado (expoente/17) — magnitude relativa contínua.
  - 19    : expoente máximo do board (broadcast, /17) — magnitude absoluta / estágio.
"""

from __future__ import annotations

import numpy as np

from .board import GameState

NUM_CHANNELS = 20
MAX_EXPONENT = 17  # teto do one-hot (2^17); folga generosa


def encode(state: GameState) -> np.ndarray:
    """Retorna array float32 de forma (20, n, n)."""
    n = state.size
    cells = np.asarray(state.cells, dtype=np.int64).reshape(n, n)
    x = np.zeros((NUM_CHANNELS, n, n), dtype=np.float32)

    # one-hot expoentes 1..17 (satura expoentes acima de 17 no último canal)
    clipped = np.clip(cells, 0, MAX_EXPONENT)
    for e in range(1, MAX_EXPONENT + 1):
        x[e - 1] = clipped == e

    x[17] = cells == 0
    x[18] = cells.astype(np.float32) / MAX_EXPONENT
    x[19] = float(cells.max()) / MAX_EXPONENT if cells.size else 0.0
    return x


def encode_batch(states: list[GameState]) -> np.ndarray:
    """Empilha estados de MESMO tamanho em (B, 20, n, n). (Batches são por-tamanho:
    dentro de uma busca/partida todos os boards têm o mesmo n; multi-tamanho só
    ocorre entre partidas.)"""
    if not states:
        return np.zeros((0, NUM_CHANNELS, 0, 0), dtype=np.float32)
    n = states[0].size
    assert all(s.size == n for s in states), "encode_batch exige tamanho uniforme"
    return np.stack([encode(s) for s in states], axis=0)

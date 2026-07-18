"""Replay buffer — janela FIFO de posições recentes, agrupadas por tamanho.

Multi-tamanho: um lote de treino precisa ter n uniforme (convolução), então as
posições ficam em ring buffers por tamanho e a amostragem escolhe um tamanho e
tira o lote dele. Cada posição: (board, alvo de política (4,), alvo de valor).
"""

from __future__ import annotations

import numpy as np

from .board import GameState


class _Ring:
    __slots__ = ("cap", "data", "pos")

    def __init__(self, cap: int):
        self.cap = cap
        self.data: list = []
        self.pos = 0

    def append(self, item) -> None:
        if len(self.data) < self.cap:
            self.data.append(item)
        else:
            self.data[self.pos] = item
            self.pos = (self.pos + 1) % self.cap

    def __len__(self) -> int:
        return len(self.data)


class ReplayBuffer:
    def __init__(self, capacity_per_size: int = 100_000):
        self.cap = capacity_per_size
        self.by_size: dict[int, _Ring] = {}

    def add(self, size: int, state: GameState, policy: np.ndarray, value: float) -> None:
        ring = self.by_size.get(size)
        if ring is None:
            ring = self.by_size[size] = _Ring(self.cap)
        ring.append((state, policy.astype(np.float32), np.float32(value)))

    def total(self) -> int:
        return sum(len(r) for r in self.by_size.values())

    def sizes_ready(self, min_count: int) -> list[int]:
        return [s for s, r in self.by_size.items() if len(r) >= min_count]

    def sample(
        self, batch_size: int, rng: np.random.Generator, size: int
    ) -> tuple[list[GameState], np.ndarray, np.ndarray]:
        ring = self.by_size[size]
        idx = rng.integers(0, len(ring), size=batch_size)
        items = [ring.data[i] for i in idx]
        states = [it[0] for it in items]
        policies = np.stack([it[1] for it in items]).astype(np.float32)
        values = np.array([it[2] for it in items], dtype=np.float32)
        return states, policies, values

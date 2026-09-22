"""Testes do pool de posições iniciais (self-play de endgame)."""

import json

import numpy as np

from twenty48.parallel import _make_slots, load_start_pool


def _write_pool(tmp_path):
    positions = [
        {"size": 4, "cells": [10, 1, 2, 3, 4, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0], "score": 12000},
        {"size": 4, "cells": [1, 2, 10, 3, 0, 1, 2, 4, 0, 0, 1, 2, 0, 0, 0, 1], "score": 9000},
        {"size": 5, "cells": [10] + [1] * 24, "score": 15000},
    ]
    p = tmp_path / "pool.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in positions))
    return p


def test_load_start_pool(tmp_path):
    pool = load_start_pool(_write_pool(tmp_path))
    assert set(pool) == {4, 5}
    assert len(pool[4]) == 2 and len(pool[5]) == 1
    st = pool[4][0]
    assert st.size == 4 and st.score == 12000 and st.cells[0] == 10  # tile 1024 no canto


def test_make_slots_samples_from_pool(tmp_path):
    pool = load_start_pool(_write_pool(tmp_path))
    rng = np.random.default_rng(0)
    slots = _make_slots(8, (4,), rng, start_pool=pool)
    pool_scores = {12000, 9000}
    for s in slots:
        assert s.size == 4
        # começou de uma posição do pool (não do tabuleiro vazio, score 0)
        assert s.state.score in pool_scores
        assert s.state.cells in {p.cells for p in pool[4]}


def test_make_slots_empty_board_without_pool():
    rng = np.random.default_rng(0)
    slots = _make_slots(4, (4,), rng, start_pool=None)
    for s in slots:
        assert s.state.score == 0  # tabuleiro inicial

"""Paridade do motor Python contra fixtures geradas pelo lado TS.

Blinda contra divergência silenciosa entre a busca de treino (Python) e a que
roda no browser (TS) — o bug mais difícil do projeto. Regenerar fixtures:
`cd play && npm run gen:fixtures`.
"""

import json
import math
from pathlib import Path

import pytest

from twenty48.board import (
    ACTIONS,
    GameState,
    apply_move,
    is_terminal,
    legal_actions,
    normalize_score,
    spawn_outcomes,
)

FIX = Path(__file__).parent / "fixtures" / "engine_parity.json"
CASES = json.loads(FIX.read_text())["cases"]


def _state(case) -> GameState:
    return GameState(size=case["size"], cells=tuple(case["cells"]), score=case["score"])


@pytest.mark.parametrize("case", CASES, ids=[f"{i}-n{c['size']}" for i, c in enumerate(CASES)])
def test_apply_move_parity(case):
    state = _state(case)
    for a in ACTIONS:
        expected = case["moves"][str(a)]
        cells, gained, moved = apply_move(state, a)
        assert list(cells) == expected["cells"], f"cells mismatch action={a}"
        assert gained == expected["gained"], f"gained mismatch action={a}"
        assert moved == expected["moved"], f"moved mismatch action={a}"


@pytest.mark.parametrize("case", CASES, ids=[f"{i}-n{c['size']}" for i, c in enumerate(CASES)])
def test_terminal_and_legal_parity(case):
    state = _state(case)
    assert is_terminal(state) == case["terminal"]
    assert legal_actions(state) == case["legal"]


@pytest.mark.parametrize("case", CASES, ids=[f"{i}-n{c['size']}" for i, c in enumerate(CASES)])
def test_spawn_outcomes_parity(case):
    got = spawn_outcomes(tuple(case["cells"]))
    expected = case["spawnOutcomes"]
    assert len(got) == len(expected)
    total = 0.0
    for g, e in zip(got, expected):
        assert g.index == e["index"]
        assert g.exponent == e["exponent"]
        assert math.isclose(g.prob, e["prob"], rel_tol=1e-12, abs_tol=1e-12)
        total += g.prob
    if expected:
        assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_normalize_score_parity():
    for case in CASES:
        assert math.isclose(
            normalize_score(case["score"]), case["normalizedScore"], rel_tol=1e-12, abs_tol=1e-12
        )

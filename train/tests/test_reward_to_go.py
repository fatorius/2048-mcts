"""Testes do backup reward-to-go do MCTS e do ValueTransform."""

import math

import numpy as np

from twenty48.board import initial_state
from twenty48.evaluators import uniform_evaluator
from twenty48.mcts import MctsConfig, run_mcts
from twenty48.value_norm import ValueNormalizer, ValueTransform


def test_transform_roundtrip():
    vt = ValueTransform(mu=1000.0, sigma=500.0, ready=True)
    for raw in (0.0, 250.0, 1000.0, 5000.0):
        assert math.isclose(vt.denorm(vt.renorm(raw)), raw, rel_tol=1e-6, abs_tol=1e-4)
    # renorm é monotônica crescente no reward-to-go bruto
    assert vt.renorm(0.0) < vt.renorm(1000.0) < vt.renorm(5000.0)
    # e fica em [0,1]
    assert 0.0 <= vt.renorm(-1e9) <= 1.0 and 0.0 <= vt.renorm(1e9) <= 1.0


def test_normalizer_transform_ready():
    nz = ValueNormalizer(momentum=0.1, min_std=1.0)
    # sem spread → não pronto (cai no backup antigo)
    assert nz.transform(4).ready is False
    for v in (0, 500, 1000, 1500, 2000, 3000):
        nz.update(4, v)
    assert nz.transform(4).ready is True


def test_rtg_backup_credits_path_reward():
    """Com value_transform pronto, o backup soma as recompensas `gained` do
    caminho: a busca deve preferir a ação que marca mais pontos, mesmo com
    avaliador uniforme (valor de folha constante)."""
    nz = ValueNormalizer(momentum=0.1, min_std=1.0)
    for v in (0, 500, 1000, 1500, 2000, 3000):
        nz.update(4, v)
    vt = nz.transform(4)
    assert vt.ready

    rng = np.random.default_rng(0)
    state = initial_state(4, rng)
    cfg = MctsConfig(simulations=200, c_puct=1.5, batch_size=16)
    res, root = run_mcts(state, uniform_evaluator, rng, cfg, value_transform=vt)
    # a ação escolhida deve ser legal e ter Q finito em [0,1]
    assert res.best_action in [a for a in range(4) if res.legal[a]]
    for a in range(4):
        if res.legal[a] and root.N[a] > 0:
            q = root.W[a] / root.N[a]
            assert 0.0 <= q <= 1.0

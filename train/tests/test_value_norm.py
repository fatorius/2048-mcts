"""O alvo de valor padronizado deve ter range útil (o bug do primeiro run)."""

import numpy as np

from twenty48.board import normalize_score
from twenty48.value_norm import ValueNormalizer


def test_returns_half_before_spread():
    vn = ValueNormalizer()
    assert vn.normalize(1000, 4) == 0.5  # sem dados
    vn.update(4, 1000)
    assert vn.normalize(1000, 4) == 0.5  # sem variância ainda


def test_spread_beats_log2_normalization():
    # Distribuição realista de scores 4×4 do primeiro run (~800–1900).
    rng = np.random.default_rng(0)
    scores = rng.normal(1200, 350, size=4000).clip(200, 6000)
    vn = ValueNormalizer(momentum=0.02)
    for s in scores:
        vn.update(4, s)

    z_new = vn.normalize_array(scores, 4)
    z_old = np.array([normalize_score(s) for s in scores])

    # Padronização espalha muito mais que log2/16 (que comprimia ~0.05).
    assert z_new.std() > 5 * z_old.std()
    assert z_new.min() < 0.2 and z_new.max() > 0.8
    assert z_new.min() >= 0.0 and z_new.max() <= 1.0
    # Monotônico: score maior -> valor maior.
    lo = vn.normalize(600, 4)
    hi = vn.normalize(2000, 4)
    assert hi > lo


def test_state_dict_roundtrip():
    import json

    rng = np.random.default_rng(2)
    vn = ValueNormalizer(momentum=0.05)
    for _ in range(300):
        vn.update(4, float(rng.normal(1200, 300)))
        vn.update(6, float(rng.normal(40000, 8000)))

    # Round-trip com chaves int (torch.save) e com chaves str (via JSON).
    for state in (vn.state_dict(), json.loads(json.dumps(vn.state_dict()))):
        vn2 = ValueNormalizer()
        vn2.load_state_dict(state)
        assert vn2.normalize(1500, 4) == vn.normalize(1500, 4)
        assert vn2.normalize(45000, 6) == vn.normalize(45000, 6)


def test_per_size_independent():
    rng = np.random.default_rng(1)
    vn = ValueNormalizer(momentum=0.05)
    for _ in range(400):
        vn.update(4, float(rng.normal(1200, 300)))  # 4×4 ~1200
        vn.update(6, float(rng.normal(40000, 8000)))  # 6×6 ~40000 (escala diferente)
    # Um score de 3000 é ótimo para 4×4, mas baixíssimo relativo a 6×6.
    assert vn.normalize(3000, 4) > 0.9
    assert vn.normalize(3000, 6) < 0.1


def test_recalibrate_order_independent():
    import numpy as np
    from twenty48.value_norm import ValueNormalizer
    vals = np.arange(0, 20000, 7, dtype=np.float64)  # rtg realista, espalhado
    # ordem crescente vs decrescente vs embaralhada -> mesmo μ,σ (sem viés)
    mus, sigs = [], []
    for order in (vals, vals[::-1], np.random.default_rng(0).permutation(vals)):
        nz = ValueNormalizer()
        nz.recalibrate(4, order)
        mu, sig = nz._mu_sigma(4)
        mus.append(mu); sigs.append(sig)
    assert max(mus) - min(mus) < 1e-6
    assert max(sigs) - min(sigs) < 1e-6
    # μ ≈ média real (ao contrário do EMA por-posição enviesado)
    assert abs(mus[0] - vals.mean()) < 1.0
    # e o valor NÃO satura: rtg alto e baixo mapeiam para valores distintos
    nz = ValueNormalizer(); nz.recalibrate(4, vals)
    assert nz.normalize(2000, 4) < nz.normalize(18000, 4) - 0.1


def test_buffer_values_roundtrip():
    import numpy as np
    from twenty48.buffer import ReplayBuffer
    from twenty48.board import GameState
    buf = ReplayBuffer(1000)
    for r in (100.0, 5000.0, 20000.0):
        buf.add(4, GameState(4, tuple([0]*16), 0), np.full(4, 0.25, np.float32), r)
    v = buf.values(4)
    assert sorted(v.tolist()) == [100.0, 5000.0, 20000.0]

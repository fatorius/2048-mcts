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


def test_per_size_independent():
    rng = np.random.default_rng(1)
    vn = ValueNormalizer(momentum=0.05)
    for _ in range(400):
        vn.update(4, float(rng.normal(1200, 300)))  # 4×4 ~1200
        vn.update(6, float(rng.normal(40000, 8000)))  # 6×6 ~40000 (escala diferente)
    # Um score de 3000 é ótimo para 4×4, mas baixíssimo relativo a 6×6.
    assert vn.normalize(3000, 4) > 0.9
    assert vn.normalize(3000, 6) < 0.1

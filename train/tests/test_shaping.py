"""Shaping posicional: cantos positivos, centro negativo, proporcional ao valor."""
import numpy as np

from twenty48.board import GameState
from twenty48.shaping import cell_weights, positional_bonus


def test_cell_weights_4x4_counts():
    w = cell_weights(4, 1.0, 0.25, -1.0)
    assert (w == 1.0).sum() == 4      # 4 cantos
    assert (w == 0.25).sum() == 8     # 8 bordas
    assert (w == -1.0).sum() == 4     # 4 centro
    # cantos são as posições certas
    for i in (0, 3, 12, 15):
        assert w[i] == 1.0
    for i in (5, 6, 9, 10):
        assert w[i] == -1.0


def test_bonus_corner_positive_center_negative():
    w = cell_weights(4, 1.0, 0.25, -1.0)
    # peça grande só num canto -> bônus = +1 (só ela conta)
    corner = GameState(4, tuple([11] + [0] * 15), 0)
    center = GameState(4, tuple([0]*5 + [11] + [0]*10), 0)  # célula 5 = centro
    b = positional_bonus([corner, center], w)
    assert b[0] > 0.9        # canto -> ~+1
    assert b[1] < -0.9       # centro -> ~-1


def test_bonus_proportional_to_tile_value():
    w = cell_weights(4, 1.0, 0.25, -1.0)
    # canto grande + centro pequeno -> domina o canto (proporcional a 2^exp)
    s = GameState(4, tuple([11, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0)
    b = positional_bonus([s], w)[0]
    # 2^11 no canto (+1) vs 2^1 no centro (-1): (2048 - 2)/(2050) ~ +0.998
    assert b > 0.99


def test_empty_board_zero():
    w = cell_weights(4, 1.0, 0.25, -1.0)
    assert positional_bonus([GameState(4, tuple([0]*16), 0)], w)[0] == 0.0

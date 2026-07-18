import numpy as np

from twenty48.board import GameState
from twenty48.evaluators import uniform_evaluator
from twenty48.mcts import MctsConfig, run_mcts, select_move


def state(grid, score=0):
    size = len(grid)
    cells = tuple(grid[r][c] for r in range(size) for c in range(size))
    return GameState(size, cells, score)


def test_visits_conserved_and_only_legal():
    s = state([[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    cfg = MctsConfig(simulations=256, batch_size=16, c_puct=1.5)
    res, _ = run_mcts(s, uniform_evaluator, np.random.default_rng(0), cfg)
    # Toda descida passa por exatamente uma aresta da raiz => soma == simulações.
    assert sum(res.visits) == cfg.simulations
    for a in range(4):
        if not res.legal[a]:
            assert res.visits[a] == 0
    assert res.best_action >= 0 and res.legal[res.best_action]


def test_terminal_returns_no_move():
    s = state([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]])
    res, _ = run_mcts(s, uniform_evaluator, np.random.default_rng(1), MctsConfig(simulations=32))
    assert res.best_action == -1


def test_deterministic_given_seed():
    s = state([[1, 1, 2, 0], [0, 2, 0, 0], [3, 0, 0, 1], [0, 0, 0, 0]])
    cfg = MctsConfig(simulations=300, batch_size=8)
    a, _ = run_mcts(s, uniform_evaluator, np.random.default_rng(7), cfg)
    b, _ = run_mcts(s, uniform_evaluator, np.random.default_rng(7), cfg)
    assert a.visits == b.visits
    assert a.best_action == b.best_action


def test_batch_size_conserves_total():
    # Qualquer batch_size preserva o total de visitas (invariante de ondas).
    s = state([[1, 2, 3, 0], [0, 1, 0, 2], [0, 0, 0, 0], [1, 0, 0, 0]])
    for bs in (1, 8, 64):
        cfg = MctsConfig(simulations=128, batch_size=bs)
        res, _ = run_mcts(s, uniform_evaluator, np.random.default_rng(3), cfg)
        assert sum(res.visits) == 128


def test_select_move_temperature():
    rng = np.random.default_rng(0)
    assert select_move([10, 0, 5, 0], 0.0, rng) == 0  # argmax
    assert select_move([0, 0, 0, 0], 1.0, rng) == -1  # sem visitas
    # τ=1 amostra proporcional: com massa só em índice 2, sempre retorna 2
    assert select_move([0, 0, 7, 0], 1.0, rng) == 2

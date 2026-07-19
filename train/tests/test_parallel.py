import numpy as np

from twenty48.board import GameState, normalize_score
from twenty48.buffer import ReplayBuffer
from twenty48.evaluators import uniform_evaluator
from twenty48.mcts import MctsConfig, mcts_search_gen, run_mcts
from twenty48.parallel import evaluate_parallel, play_games_parallel
from twenty48.value_norm import ValueNormalizer


def _signal_eval(states):
    """Avaliador determinístico COM sinal (política não-uniforme + valor variável),
    para que a busca sequencial e a em lote de fato divirjam."""
    b = len(states)
    pols = np.tile(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32), (b, 1))
    vals = np.array([min(s.score / 20000.0, 1.0) for s in states], dtype=np.float32)
    return pols, vals


def _drive(gen, evaluator):
    try:
        s = next(gen)
        while True:
            pol, val = evaluator([s])
            s = gen.send((pol[0], val[0]))
    except StopIteration as e:
        return e.value


def _state():
    return GameState(4, (1, 1, 2, 0, 0, 2, 0, 0, 3, 0, 0, 1, 0, 0, 0, 4), 5000)


def test_generator_matches_sequential_run_mcts():
    # A corrotina sequencial == run_mcts(batch_size=1), exatamente (mesmo rng/eval).
    cfg = MctsConfig(simulations=400, c_puct=1.5, batch_size=1)
    r_gen = _drive(mcts_search_gen(_state(), cfg, np.random.default_rng(3), add_noise=False), _signal_eval)
    r_seq, _ = run_mcts(_state(), _signal_eval, np.random.default_rng(3), cfg)
    assert r_gen.visits == r_seq.visits
    assert r_gen.best_action == r_seq.best_action


def test_generator_differs_from_batched():
    # Com sinal, a busca em lote (batch=32, virtual loss) diverge da sequencial.
    cfg1 = MctsConfig(simulations=400, c_puct=1.5, batch_size=1)
    r_gen = _drive(mcts_search_gen(_state(), cfg1, np.random.default_rng(3), add_noise=False), _signal_eval)
    r_b32, _ = run_mcts(_state(), _signal_eval, np.random.default_rng(3), MctsConfig(400, 1.5, 32))
    assert r_gen.visits != r_b32.visits


def test_play_games_parallel_fills_buffer():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer()
    norm = ValueNormalizer()
    cfg = MctsConfig(simulations=32, c_puct=1.5)
    stats = play_games_parallel(4, (4,), uniform_evaluator, rng, cfg, buf, norm, temp_moves=10, move_cap=200)
    assert len(stats) == 4
    assert all(s.size == 4 for s in stats)
    assert all(s.moves > 0 for s in stats)
    assert buf.total() > 0  # posições gravadas
    assert 4 in norm._mean  # normalizador atualizado


def test_evaluate_parallel_runs():
    rng = np.random.default_rng(1)
    m = evaluate_parallel(uniform_evaluator, rng, size=4, games=4, sims=32, move_cap=200)
    assert m.games == 4
    assert m.mean_score > 0
    assert 0.0 <= m.reach_2048_rate <= 1.0

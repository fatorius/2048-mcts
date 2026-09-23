"""Encoding + rede: forma, faixa de valor e agnosticismo de tamanho."""

import numpy as np
import torch

from twenty48.board import GameState
from twenty48.encode import NUM_CHANNELS, encode, encode_batch
from twenty48.net import NUM_ACTIONS, Net, param_count


def _state(size, fill=0.5, seed=0):
    rng = np.random.default_rng(seed)
    cells = tuple(int(1 + rng.integers(0, 6)) if rng.random() < fill else 0 for _ in range(size * size))
    return GameState(size=size, cells=cells, score=0)


def test_encode_shape_and_channels():
    s = _state(4)
    x = encode(s)
    assert x.shape == (NUM_CHANNELS, 4, 4)
    assert x.dtype == np.float32
    # canal 17 = vazio; complementa os ocupados
    empties = sum(1 for v in s.cells if v == 0)
    assert x[17].sum() == empties


def test_encode_onehot_matches_exponent():
    s = GameState(4, (0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 0)
    x = encode(s)
    # célula (0,1) tem expoente 1 -> canal 0 ligado ali
    assert x[0, 0, 1] == 1.0
    assert x[1, 0, 2] == 1.0  # expoente 2 -> canal 1
    assert x[2, 0, 3] == 1.0  # expoente 3 -> canal 2
    # célula vazia (0,0): nenhum canal one-hot ligado
    assert x[:17, 0, 0].sum() == 0.0
    # max exponent broadcast = 3/17
    assert np.allclose(x[19], 3 / 17)


def test_net_forward_size_agnostic():
    net = Net(channels=32, blocks=3).eval()
    for n in (3, 4, 5, 6):
        states = [_state(n, seed=i) for i in range(4)]
        x = torch.from_numpy(encode_batch(states))
        with torch.no_grad():
            logits, value = net(x)
        assert logits.shape == (4, NUM_ACTIONS)
        assert value.shape == (4,)
        assert torch.all(value >= 0) and torch.all(value <= 1)


def test_net_new_arch_size_agnostic():
    # Arquitetura nova (coord + avgmax pool + policy hidden): mesma interface,
    # continua agnóstica ao tamanho e valor em [0,1].
    net = Net(channels=32, blocks=3, coord=True, pool="avgmax", policy_hidden=True).eval()
    for n in (3, 4, 5, 6):
        states = [_state(n, seed=i) for i in range(4)]
        x = torch.from_numpy(encode_batch(states))
        with torch.no_grad():
            logits, value = net(x)
        assert logits.shape == (4, NUM_ACTIONS)
        assert value.shape == (4,)
        assert torch.all(value >= 0) and torch.all(value <= 1)


def test_net_coord_breaks_translation_invariance():
    # CoordConv deve tornar a rede sensível à POSIÇÃO: o mesmo tile em cantos
    # diferentes produz saídas diferentes (o GAP puro daria idêntico).
    torch.manual_seed(0)
    net = Net(channels=16, blocks=2, coord=True, pool="avgmax", policy_hidden=True).eval()
    a = GameState(4, (1,) + (0,) * 15, 0)               # tile no canto sup-esq
    b = GameState(4, (0,) * 15 + (1,), 0)               # tile no canto inf-dir
    xa = torch.from_numpy(encode_batch([a]))
    xb = torch.from_numpy(encode_batch([b]))
    with torch.no_grad():
        la, _ = net(xa)
        lb, _ = net(xb)
    assert not torch.allclose(la, lb, atol=1e-5)


def test_net_default_is_legacy_arch():
    # Default = arquitetura antiga (p/ carregar checkpoints legados): input 20 canais
    # (sem coord), pool avg (feat = channels), política Linear cru.
    net = Net(channels=32, blocks=2)
    assert net.trunk[0].in_channels == NUM_CHANNELS
    assert isinstance(net.policy_head, torch.nn.Linear)
    assert net.value_head[0].in_features == 32  # avg pool → feat = channels


def test_net_same_weights_all_sizes():
    # A MESMA instância (mesmos pesos) processa qualquer n — nenhum parâmetro
    # depende da posição/tamanho.
    net = Net(channels=32, blocks=3).eval()
    x3 = torch.from_numpy(encode_batch([_state(3, seed=1)]))
    x6 = torch.from_numpy(encode_batch([_state(6, seed=1)]))
    with torch.no_grad():
        net(x3)
        net(x6)
    assert param_count(net) > 0

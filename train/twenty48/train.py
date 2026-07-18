"""Orquestrador do loop de RL (self-play + treino), estilo AlphaZero.

Um processo, um laço: gera partidas com MCTS guiado pela rede → replay buffer →
passos de gradiente (valor MSE + política CE + L2) → rede melhor → dados melhores.
Warm start: a iteração 0 gera dados com o MCTS-rollout da Fase 1 (a rede recém-
inicializada tem value head de ruído).

Uso:
  python -m twenty48.train --smoke          # valida o pipeline (rápido)
  python -m twenty48.train --iterations 40  # treino de verdade
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .encode import encode_batch
from .evaluate import evaluate_net
from .evaluators import NetEvaluator, RolloutEvaluator
from .export_onnx import export_onnx
from .mcts import MctsConfig
from .net import Net, param_count
from .self_play import self_play_game

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


@dataclass
class TrainConfig:
    sizes: tuple[int, ...] = (4,)
    iterations: int = 40
    games_per_iter: int = 24
    move_cap: int = 4000
    temp_moves: int = 20
    sims: int = 100
    c_puct: float = 1.5
    mcts_batch: int = 32
    rollout_n: int = 1
    warm_start: bool = True
    train_steps: int = 200
    train_batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    buffer_per_size: int = 200_000
    channels: int = 64
    blocks: int = 4
    eval_size: int = 4
    eval_games: int = 12
    eval_sims: int = 100
    seed: int = 0


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _mean(xs, key):
    return float(np.mean([key(x) for x in xs])) if xs else 0.0


def train(cfg: TrainConfig) -> None:
    device = pick_device()
    CKPT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    net = Net(cfg.channels, cfg.blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.buffer_per_size)
    mcts_cfg = MctsConfig(simulations=cfg.sims, c_puct=cfg.c_puct, batch_size=cfg.mcts_batch)

    print(f"device={device}  params={param_count(net)}  sizes={cfg.sizes}")
    best_score = -1.0

    for it in range(cfg.iterations):
        # --- 1. SELF-PLAY ---
        use_rollout = it == 0 and cfg.warm_start
        net.eval()
        evaluator = (
            RolloutEvaluator(rng, cfg.rollout_n) if use_rollout else NetEvaluator(net, device)
        )
        t0 = time.time()
        sp_stats = []
        for _ in range(cfg.games_per_iter):
            size = int(rng.choice(cfg.sizes))
            sp_stats.append(
                self_play_game(evaluator, rng, mcts_cfg, size, buffer, cfg.temp_moves, cfg.move_cap)
            )
        sp_time = time.time() - t0

        # --- 2. TRAIN ---
        net.train()
        ready = buffer.sizes_ready(cfg.train_batch)
        loss_acc = np.zeros(3)
        n_steps = 0
        if ready:
            for _ in range(cfg.train_steps):
                size = int(rng.choice(ready))
                states, pol, val = buffer.sample(cfg.train_batch, rng, size)
                x = torch.from_numpy(encode_batch(states)).to(device)
                target_p = torch.from_numpy(pol).to(device)
                target_v = torch.from_numpy(val).to(device)
                logits, value = net(x)
                loss_v = F.mse_loss(value, target_v)
                loss_p = -(target_p * F.log_softmax(logits, dim=1)).sum(1).mean()
                loss = loss_v + loss_p
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_acc += [loss.item(), loss_v.item(), loss_p.item()]
                n_steps += 1
        avg = loss_acc / max(1, n_steps)

        # --- 3. EVAL + CHECKPOINT ---
        m = evaluate_net(
            net, device, rng, cfg.eval_size, cfg.eval_games, cfg.eval_sims, cfg.c_puct,
            cfg.mcts_batch, cfg.move_cap,
        )
        tag = "rollout" if use_rollout else "net"
        print(
            f"[it {it:02d}] self-play({tag}) {cfg.games_per_iter}g "
            f"score~{_mean(sp_stats, lambda s: s.score):.0f} "
            f"tile~{1 << round(_mean(sp_stats, lambda s: s.max_exponent))} {sp_time:.0f}s | "
            f"buf={buffer.total()} steps={n_steps} loss={avg[0]:.3f}(v{avg[1]:.3f}/p{avg[2]:.3f}) | "
            f"EVAL score={m.mean_score:.0f} 2048={m.reach_2048_rate:.0%} best={m.best_tile}"
        )

        ckpt = {"net": net.state_dict(), "cfg": asdict(cfg), "iter": it, "eval_mean_score": m.mean_score}
        torch.save(ckpt, CKPT_DIR / "latest.pt")
        if m.mean_score > best_score:
            best_score = m.mean_score
            torch.save(ckpt, CKPT_DIR / "best.pt")

    onnx_path = CKPT_DIR / "model.onnx"
    export_onnx(net, str(onnx_path), example_size=cfg.sizes[0])
    print(f"exported ONNX -> {onnx_path}")


def _parse() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="run tiny config to validate pipeline")
    p.add_argument("--iterations", type=int)
    p.add_argument("--games-per-iter", type=int)
    p.add_argument("--sims", type=int)
    p.add_argument("--train-steps", type=int)
    p.add_argument("--sizes", type=int, nargs="+")
    p.add_argument("--eval-games", type=int)
    p.add_argument("--eval-sims", type=int)
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    cfg = TrainConfig()
    if args.smoke:
        cfg = TrainConfig(
            iterations=2, games_per_iter=2, move_cap=250, sims=16, mcts_batch=8,
            train_steps=10, train_batch=32, eval_games=2, eval_sims=16, channels=32, blocks=3,
        )
    for k in ("iterations", "sims", "train_steps", "seed"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.games_per_iter is not None:
        cfg.games_per_iter = args.games_per_iter
    if args.sizes is not None:
        cfg.sizes = tuple(args.sizes)
    if args.eval_games is not None:
        cfg.eval_games = args.eval_games
    if args.eval_sims is not None:
        cfg.eval_sims = args.eval_sims
    return cfg


if __name__ == "__main__":
    train(_parse())

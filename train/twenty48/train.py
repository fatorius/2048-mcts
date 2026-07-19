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
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
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
from .value_norm import ValueNormalizer

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


RESUME_LATEST = "__latest__"


def _latest_run_dir() -> Path:
    runs = sorted(CKPT_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"nenhum run_* em {CKPT_DIR} para retomar")
    return runs[0]


def _resolve_resume(path: str) -> Path:
    """Aceita RESUME_LATEST (run mais recente), um .pt direto, ou uma pasta de run
    (usa best.pt dela)."""
    p = _latest_run_dir() if path == RESUME_LATEST else Path(path)
    if p.is_dir():
        p = p / "best.pt"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint não encontrado: {p}")
    return p


def train(cfg: TrainConfig, resume: str | None = None) -> None:
    device = pick_device()
    # Resolve o resume ANTES de criar o novo run_dir (senão o novo, recém-criado,
    # seria o "mais recente").
    ckpt_path = _resolve_resume(resume) if resume else None

    # Diretório autocontido por run: config + log de métricas + checkpoints + onnx.
    run_dir = CKPT_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Carrega o checkpoint ANTES de construir a rede — a arquitetura precisa casar
    # com os pesos salvos.
    ckpt = None
    resumed_from = None
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ckpt.get("cfg", {})
        cfg.channels = saved.get("channels", cfg.channels)
        cfg.blocks = saved.get("blocks", cfg.blocks)
        cfg.warm_start = False  # já temos rede treinada — sem rollout na it0
        resumed_from = str(ckpt_path)

    net = Net(cfg.channels, cfg.blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.buffer_per_size)
    normalizer = ValueNormalizer()
    best_score = -1.0

    if ckpt is not None:
        net.load_state_dict(ckpt["net"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "normalizer" in ckpt:
            normalizer.load_state_dict(ckpt["normalizer"])
        best_score = ckpt.get("eval_mean_score", -1.0)  # não regride o best
        # Semeia o best.pt do novo run com o modelo carregado, para a pasta ficar
        # autocontida mesmo que nenhuma iteração supere o best histórico.
        torch.save(ckpt, run_dir / "best.pt")
        print(
            f"resumido de {resumed_from} (iter orig {ckpt.get('iter')}, eval {best_score:.0f}) | "
            f"opt={'ok' if 'opt' in ckpt else 'novo'} "
            f"normalizer={'ok' if 'normalizer' in ckpt else 'novo'} buffer=novo (não persistido)",
            flush=True,
        )

    (run_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "device": device, "resumed_from": resumed_from}, indent=2)
    )
    mcts_cfg = MctsConfig(simulations=cfg.sims, c_puct=cfg.c_puct, batch_size=cfg.mcts_batch)

    print(f"device={device}  params={param_count(net)}  sizes={cfg.sizes}", flush=True)
    print(f"run dir: {run_dir}", flush=True)

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
                self_play_game(
                    evaluator, rng, mcts_cfg, size, buffer, normalizer, cfg.temp_moves, cfg.move_cap
                )
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
                states, pol, raw_scores = buffer.sample(cfg.train_batch, rng, size)
                x = torch.from_numpy(encode_batch(states)).to(device)
                target_p = torch.from_numpy(pol).to(device)
                # Alvo de valor padronizado com os μ,σ correntes (bem espalhado em [0,1]).
                target_v = torch.from_numpy(normalizer.normalize_array(raw_scores, size)).to(device)
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
            cfg.mcts_batch, cfg.move_cap, normalizer,
        )
        tag = "rollout" if use_rollout else "net"
        print(
            f"[it {it:02d}] self-play({tag}) {cfg.games_per_iter}g "
            f"score~{_mean(sp_stats, lambda s: s.score):.0f} "
            f"tile~{1 << round(_mean(sp_stats, lambda s: s.max_exponent))} {sp_time:.0f}s | "
            f"buf={buffer.total()} steps={n_steps} loss={avg[0]:.3f}(v{avg[1]:.3f}/p{avg[2]:.3f}) | "
            f"EVAL score={m.mean_score:.0f} 2048={m.reach_2048_rate:.0%} best={m.best_tile}",
            flush=True,
        )

        # Registro estruturado por iteração (uma linha JSON; flush → sobrevive a
        # crash e é legível ao vivo mesmo com stdout buferizado).
        record = {
            "iter": it,
            "phase": tag,
            "selfplay_mean_score": _mean(sp_stats, lambda s: s.score),
            "selfplay_mean_max_exponent": _mean(sp_stats, lambda s: s.max_exponent),
            "selfplay_best_tile": 1 << max((s.max_exponent for s in sp_stats), default=0),
            "selfplay_seconds": sp_time,
            "buffer_total": buffer.total(),
            "train_steps": n_steps,
            "loss": avg[0],
            "loss_value": avg[1],
            "loss_policy": avg[2],
            "eval_mean_score": m.mean_score,
            "eval_best_tile": m.best_tile,
            "eval_reach_2048": m.reach_2048_rate,
            "eval_reach_4096": m.reach_4096_rate,
            "eval_tile_hist": m.tile_hist,
        }
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        ckpt = {
            "net": net.state_dict(),
            "opt": opt.state_dict(),
            "normalizer": normalizer.state_dict(),
            "cfg": asdict(cfg),
            "iter": it,
            "eval_mean_score": m.mean_score,
        }
        torch.save(ckpt, run_dir / "latest.pt")
        if m.mean_score > best_score:
            best_score = m.mean_score
            torch.save(ckpt, run_dir / "best.pt")

    onnx_path = run_dir / "model.onnx"
    export_onnx(net, str(onnx_path), example_size=cfg.sizes[0])
    print(f"exported ONNX -> {onnx_path}", flush=True)
    print(f"metrics log   -> {metrics_path}", flush=True)


def _parse() -> tuple[TrainConfig, str | None]:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="run tiny config to validate pipeline")
    p.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const=RESUME_LATEST,
        help="retoma do best.pt. Sem valor = run mais recente; ou passe run dir / .pt",
    )
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
    return cfg, args.resume


if __name__ == "__main__":
    _cfg, _resume = _parse()
    train(_cfg, resume=_resume)

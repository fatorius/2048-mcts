"""Orquestrador do loop de RL (self-play + treino), estilo AlphaZero.

Um processo, um laço: gera partidas com MCTS guiado pela rede → replay buffer →
passos de gradiente (valor MSE + política CE + L2) → rede melhor → dados melhores.
Warm start: a iteração 0 gera dados com o MCTS-rollout puro (a rede recém-
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
from .evaluators import NetEvaluator, RolloutEvaluator
from .export_onnx import export_onnx
from .mcts import MctsConfig
from .net import Net, param_count
from .parallel import (
    ParallelSelfPlay,
    evaluate_parallel,
    load_start_pool,
    play_games_parallel,
)
from .value_norm import ValueNormalizer

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


@dataclass
class TrainConfig:
    sizes: tuple[int, ...] = (4,)
    iterations: int = 40
    games_per_iter: int = 24
    move_cap: int = 4000
    temp_moves: int = 20
    start_pool: str | None = None  # JSONL de posições p/ self-play de endgame (None = tabuleiro vazio)
    sims: int = 100
    c_puct: float = 1.5
    mcts_batch: int = 32
    rollout_n: int = 1
    # Warm-start com gate por eval: gera dados com o MCTS-rollout enquanto ele for
    # melhor que a rede; troca para self-play da rede quando a rede alcança
    # `warm_switch_frac` da baseline de rollout (comparadas no mesmo orçamento de
    # busca), ou no máximo em `warm_max_iters`. warm_sims=0 → usa `sims`.
    warm_start: bool = True
    warm_sims: int = 0
    warm_switch_frac: float = 0.97
    warm_max_iters: int = 1000
    train_steps: int = 200
    train_batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0  # peso do loss de valor: loss = loss_p + value_weight*loss_v
    buffer_per_size: int = 200_000
    channels: int = 128
    blocks: int = 6
    # Arquitetura (runs novos usam as melhorias por padrão; Net() mantém defaults
    # antigos p/ carregar checkpoints legados — ver net.py).
    coord: bool = True          # canais de coordenada (posição: canto/serpente)
    pool: str = "avgmax"        # average+max pooling global
    policy_hidden: bool = True  # camada oculta na cabeça de política
    eval_size: int = 4
    eval_games: int = 12
    eval_sims: int = 100
    sp_workers: int = 1  # workers de self-play (>1 = multiprocesso, lever #2)
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
        # Arquitetura DO CHECKPOINT (defaults = arq. antiga p/ checkpoints legados
        # sem essas chaves — precisam casar com os pesos salvos).
        cfg.coord = saved.get("coord", False)
        cfg.pool = saved.get("pool", "avg")
        cfg.policy_hidden = saved.get("policy_hidden", False)
        # NÃO desligamos o warm-start no resume: o gate se auto-corrige. Se a rede
        # já bater o rollout, ele abre na it0; se estiver travada abaixo (o caso de
        # querer destilar o rollout numa rede parada), ele distila até alcançar.
        # Use --no-warm-start para resumir direto em self-play da rede.
        resumed_from = str(ckpt_path)

    net = Net(
        cfg.channels, cfg.blocks,
        coord=cfg.coord, pool=cfg.pool, policy_hidden=cfg.policy_hidden,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.buffer_per_size)
    normalizer = ValueNormalizer()
    start_pool = None
    if cfg.start_pool:
        start_pool = load_start_pool(cfg.start_pool)
        print(
            "start-pool endgame: "
            + ", ".join(f"{s}×{s}:{len(v)}" for s, v in sorted(start_pool.items())),
            flush=True,
        )
    best_score = -1.0

    if ckpt is not None:
        net.load_state_dict(ckpt["net"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "normalizer" in ckpt:
            normalizer.load_state_dict(ckpt["normalizer"])
        # best.pt agora é pelo score COMBINADO (multi-tamanho); um ckpt antigo só
        # tem eval_mean_score (1 tamanho, outra escala) → começa fresco (-1).
        best_score = ckpt.get("eval_combined", -1.0)
        # Semeia o best.pt do novo run com o modelo carregado, para a pasta ficar
        # autocontida mesmo que nenhuma iteração supere o best histórico.
        torch.save(ckpt, run_dir / "best.pt")
        print(
            f"resumido de {resumed_from} (iter orig {ckpt.get('iter')}, eval {ckpt.get('eval_mean_score', -1):.0f}) | "
            f"opt={'ok' if 'opt' in ckpt else 'novo'} "
            f"normalizer={'ok' if 'normalizer' in ckpt else 'novo'} buffer=novo (não persistido)",
            flush=True,
        )

    (run_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "device": device, "resumed_from": resumed_from}, indent=2)
    )
    mcts_cfg = MctsConfig(simulations=cfg.sims, c_puct=cfg.c_puct, batch_size=cfg.mcts_batch)
    warm_sims = cfg.warm_sims or cfg.sims
    warm_mcts_cfg = MctsConfig(simulations=warm_sims, c_puct=cfg.c_puct, batch_size=cfg.mcts_batch)

    print(f"device={device}  params={param_count(net)}  sizes={cfg.sizes}", flush=True)
    print(f"run dir: {run_dir}", flush=True)

    # Gate por eval: gera dados com o MCTS-rollout enquanto ele bater a rede.
    # A baseline é medida no orçamento DO PROFESSOR (warm_sims) — é a força real
    # do rollout que gera os dados, não a de um rollout raso. Comparada contra o
    # eval da rede (eval_sims).
    using_rollout = cfg.warm_start
    rollout_baseline = 0.0
    if using_rollout:
        print(f"medindo baseline do rollout @ {warm_sims} sims ({cfg.eval_games}g)…", flush=True)
        base_t0 = time.time()
        base_done = [0]

        def _base_prog(i, sc, ex, mv):
            base_done[0] += 1
            print(
                f"  baseline {base_done[0]:>2}/{cfg.eval_games} score={sc:>6} tile={1 << ex:>5} "
                f"moves={mv:>4} ({time.time() - base_t0:.0f}s)",
                flush=True,
            )

        rollout_baseline = evaluate_parallel(
            RolloutEvaluator(rng, cfg.rollout_n), rng, cfg.eval_size, cfg.eval_games,
            warm_sims, cfg.c_puct, cfg.move_cap, on_game=_base_prog,
        ).mean_score
        print(
            f"warm-start ligado. baseline rollout (@{warm_sims} sims): {rollout_baseline:.0f} "
            f"— troca p/ rede quando eval(@{cfg.eval_sims}) ≥ {cfg.warm_switch_frac:.0%} disso "
            f"(ou it {cfg.warm_max_iters}).",
            flush=True,
        )

    # Pool de self-play multiprocesso (lever #2): sobrepõe CPU(árvore)/GPU(forward)
    # entre workers. Só p/ self-play da REDE em CUDA (o rollout do warm-start não usa
    # rede). Persistente entre iterações (CUDA inicializado 1× por worker).
    sp_pool = None
    if cfg.sp_workers > 1 and device == "cuda":
        arch = dict(channels=cfg.channels, blocks=cfg.blocks, coord=cfg.coord,
                    pool=cfg.pool, policy_hidden=cfg.policy_hidden)
        sp_pool = ParallelSelfPlay(arch, device, cfg.sp_workers)
        print(f"self-play multiprocesso: {cfg.sp_workers} workers", flush=True)

    for it in range(cfg.iterations):
        # --- 1. SELF-PLAY (rollout enquanto o gate não abrir; senão, rede) ---
        # Partidas em PARALELO com busca sequencial por partida (qualidade = TS),
        # batelando as folhas entre partidas para a GPU (parallel.py).
        net.eval()
        evaluator = RolloutEvaluator(rng, cfg.rollout_n) if using_rollout else NetEvaluator(net, device)
        sp_cfg = warm_mcts_cfg if using_rollout else mcts_cfg
        t0 = time.time()
        sp_done = [0]

        def _sp_prog(i, sc, ex, mv, _t0=t0):
            sp_done[0] += 1
            print(
                f"  sp {sp_done[0]:>2}/{cfg.games_per_iter} score={sc:>6} tile={1 << ex:>5} "
                f"moves={mv:>4} (buf={buffer.total()} {time.time() - _t0:.0f}s)",
                flush=True,
            )

        if sp_pool is not None and not using_rollout:
            # Multiprocesso: workers coletam registros e o principal agrega no
            # buffer/normalizador (sem on_game ao vivo — só self-play da rede).
            sp_stats = sp_pool.play(
                cfg.games_per_iter, cfg.sizes, net, normalizer, sp_cfg, buffer,
                cfg.temp_moves, cfg.move_cap, start_pool, base_seed=cfg.seed + it * 1000,
            )
        else:
            sp_stats = play_games_parallel(
                cfg.games_per_iter, cfg.sizes, evaluator, rng, sp_cfg, buffer, normalizer,
                cfg.temp_moves, cfg.move_cap, on_game=(_sp_prog if using_rollout else None),
                start_pool=(None if using_rollout else start_pool),
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
                loss = loss_p + cfg.value_weight * loss_v
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_acc += [loss.item(), loss_v.item(), loss_p.item()]
                n_steps += 1
        avg = loss_acc / max(1, n_steps)

        # --- 3. EVAL + CHECKPOINT (rede, gulosa, em paralelo) — POR TAMANHO ---
        # Avalia CADA tamanho treinado (do tabuleiro vazio) — um generalista tem
        # que ser medido em todas as dimensões, não só 4×4. `best.pt` pelo score
        # combinado (soma), p/ não ignorar ganhos num tamanho por perdas noutro.
        net.eval()
        evals = {}
        for esz in cfg.sizes:
            evals[esz] = evaluate_parallel(
                NetEvaluator(net, device), rng, esz, cfg.eval_games, cfg.eval_sims,
                cfg.c_puct, cfg.move_cap, terminal_value_fn=normalizer.terminal_value_fn(),
                value_transform=normalizer.transform(esz),
            )
        m = evals[cfg.sizes[0]]  # primário (gate warm-start + campos legados do report)
        # best.pt pelo score COMBINADO = soma dos LOGs por tamanho. O log normaliza
        # a escala (5×5/6×6 têm scores muito maiores que 4×4), então um ganho em
        # QUALQUER tamanho conta comparável — sem 5×5/6×6 dominar a seleção.
        combined = float(sum(np.log(e.mean_score + 1.0) for e in evals.values()))
        tag = "rollout" if using_rollout else "net"
        gate = f" [rollout base {rollout_baseline:.0f}]" if using_rollout else ""
        print(
            f"[it {it:02d}] self-play({tag}) {cfg.games_per_iter}g "
            f"score~{_mean(sp_stats, lambda s: s.score):.0f} "
            f"tile~{1 << round(_mean(sp_stats, lambda s: s.max_exponent))} {sp_time:.0f}s | "
            f"buf={buffer.total()} steps={n_steps} loss={avg[0]:.3f}(v{avg[1]:.3f}/p{avg[2]:.3f}) | "
            f"EVAL "
            + " ".join(
                f"{s}x{s}={e.mean_score:.0f}(2048={e.reach_2048_rate:.0%},t{e.best_tile})"
                for s, e in evals.items()
            )
            + gate,
            flush=True,
        )

        # --- GATE: rede alcançou o rollout? (compara no mesmo orçamento de eval) ---
        if using_rollout and (
            m.mean_score >= rollout_baseline * cfg.warm_switch_frac or it + 1 >= cfg.warm_max_iters
        ):
            why = "eval alcançou baseline" if m.mean_score >= rollout_baseline * cfg.warm_switch_frac else "warm_max_iters"
            print(
                f"  ↳ gate ABERTO ({why}): rede ({m.mean_score:.0f}) vs rollout "
                f"({rollout_baseline:.0f}). Trocando para self-play da rede.",
                flush=True,
            )
            using_rollout = False

        # Registro estruturado por iteração (uma linha JSON; flush → sobrevive a
        # crash e é legível ao vivo mesmo com stdout buferizado).
        record = {
            "iter": it,
            "phase": tag,
            "rollout_baseline": rollout_baseline,
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
            "eval_combined": combined,
            "eval_by_size": {
                str(s): {
                    "mean_score": e.mean_score,
                    "best_tile": e.best_tile,
                    "reach_2048": e.reach_2048_rate,
                    "reach_4096": e.reach_4096_rate,
                    "tile_hist": e.tile_hist,
                }
                for s, e in evals.items()
            },
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
            "eval_combined": combined,
        }
        torch.save(ckpt, run_dir / "latest.pt")
        if combined > best_score:
            best_score = combined
            torch.save(ckpt, run_dir / "best.pt")

    if sp_pool is not None:
        sp_pool.close()

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
    p.add_argument("--warm-sims", type=int, help="sims do rollout no warm-start (0 = usa --sims)")
    p.add_argument("--warm-switch-frac", type=float, help="fração da baseline p/ abrir o gate")
    p.add_argument("--warm-max-iters", type=int, help="teto de iterações de warm-start")
    p.add_argument("--no-warm-start", action="store_true", help="começa direto no self-play da rede")
    p.add_argument("--channels", type=int, help="canais da rede (treino do zero)")
    p.add_argument("--blocks", type=int, help="blocos conv da rede (treino do zero)")
    p.add_argument("--coord", action=argparse.BooleanOptionalAction, default=None,
                   help="canais de coordenada / CoordConv (treino do zero)")
    p.add_argument("--pool", choices=("avg", "avgmax"), help="pooling global (treino do zero)")
    p.add_argument("--policy-hidden", action=argparse.BooleanOptionalAction, default=None,
                   help="camada oculta na cabeça de política (treino do zero)")
    p.add_argument("--seed", type=int)
    p.add_argument("--value-weight", type=float, help="peso do loss de valor (padrão 1.0)")
    p.add_argument("--start-pool", type=str, help="JSONL de posições p/ self-play de endgame")
    p.add_argument("--sp-workers", type=int,
                   help="workers de self-play (>1 = multiprocesso; sobrepõe CPU/GPU)")
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
    if args.warm_sims is not None:
        cfg.warm_sims = args.warm_sims
    if args.warm_switch_frac is not None:
        cfg.warm_switch_frac = args.warm_switch_frac
    if args.warm_max_iters is not None:
        cfg.warm_max_iters = args.warm_max_iters
    if args.no_warm_start:
        cfg.warm_start = False
    if args.channels is not None:
        cfg.channels = args.channels
    if args.blocks is not None:
        cfg.blocks = args.blocks
    if args.coord is not None:
        cfg.coord = args.coord
    if args.pool is not None:
        cfg.pool = args.pool
    if args.policy_hidden is not None:
        cfg.policy_hidden = args.policy_hidden
    if args.value_weight is not None:
        cfg.value_weight = args.value_weight
    if args.start_pool is not None:
        cfg.start_pool = args.start_pool
    if args.sp_workers is not None:
        cfg.sp_workers = args.sp_workers
    return cfg, args.resume


if __name__ == "__main__":
    _cfg, _resume = _parse()
    train(_cfg, resume=_resume)

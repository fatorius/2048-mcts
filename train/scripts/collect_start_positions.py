"""Coleta um POOL de posições de meio/fim de jogo jogadas pela rede, para treinar
uma fase de endgame (self-play começando dessas posições em vez do tabuleiro vazio).

A rede (com o transform reward-to-go) joga partidas 4×4 e 5×5 em paralelo; grava
as posições com peça ≥ 1024 (amostradas) num JSONL: {size, cells, score}. Roda em
background, acrescentando ao arquivo. Cada posição é reconstruível como
GameState(size, tuple(cells), score).

Uso (a partir de ~/2048-mcts/train):
  PYTHONPATH=. python scripts/collect_start_positions.py \
      --ckpt checkpoints/run_20260916_084245 --out start_positions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from twenty48.board import max_exponent
from twenty48.evaluators import NetEvaluator
from twenty48.mcts import MctsConfig
from twenty48.net import Net
from twenty48.parallel import _make_slots, _run_parallel
from twenty48.value_norm import ValueNormalizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pasta de run (usa best.pt) ou .pt")
    ap.add_argument("--out", default="start_positions.jsonl")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--move-cap", type=int, default=2000, help="limita marathon p/ coleta")
    ap.add_argument("--min-exp", type=int, default=10, help="expoente mínimo (10 = tile 1024)")
    ap.add_argument("--sample-p", type=float, default=0.15, help="prob. de gravar cada posição qualificada")
    ap.add_argument("--games", type=int, default=12, help="partidas em paralelo por batch")
    ap.add_argument("--target", type=int, default=0, help="para ao atingir N posições (0 = infinito)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    p = Path(args.ckpt)
    if p.is_dir():
        p = p / "best.pt"
    data = torch.load(p, map_location=device, weights_only=False)
    cfg_saved = data.get("cfg", {})
    net = Net(cfg_saved.get("channels", 128), cfg_saved.get("blocks", 6)).to(device)
    net.load_state_dict(data["net"])
    net.eval()
    nz = ValueNormalizer()
    if data.get("normalizer"):
        nz.load_state_dict(data["normalizer"])

    ev = NetEvaluator(net, device)
    mcfg = MctsConfig(simulations=args.sims, c_puct=cfg_saved.get("c_puct", 1.5))
    tvf = nz.terminal_value_fn()
    rng = np.random.default_rng()

    outf = open(args.out, "a")
    count = [0]

    def on_finish(slot):
        for st, _pol in slot.records:
            if max_exponent(st) >= args.min_exp and rng.random() < args.sample_p:
                outf.write(json.dumps({"size": st.size, "cells": list(st.cells), "score": st.score}) + "\n")
                count[0] += 1
        outf.flush()

    print(f"coletando de {p} (device={device}) sizes={tuple(args.sizes)} sims={args.sims} "
          f"min_exp={args.min_exp} -> {args.out}", flush=True)
    batch = 0
    while True:
        slots = _make_slots(args.games, tuple(args.sizes), rng)
        _run_parallel(
            slots, ev, mcfg, tvf, True, 20, args.move_cap,
            record=True, on_finish=on_finish, transform_fn=nz.transform,
        )
        batch += 1
        print(f"batch {batch}: {count[0]} posições coletadas", flush=True)
        if args.target and count[0] >= args.target:
            print(f"alvo {args.target} atingido; parando.", flush=True)
            break


if __name__ == "__main__":
    main()

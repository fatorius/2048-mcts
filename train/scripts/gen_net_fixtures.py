"""Gera fixtures de paridade da Fase 4 a partir do lado Python (fonte da verdade):

  1. encoding: board -> tensor 20*n*n achatado (verifica o encoder TS).
  2. modelo:   board -> (policy_logits[4], value) via onnxruntime (verifica o
     caminho ORT-web fim a fim no TS).

Uso: .venv/bin/python -m scripts.gen_net_fixtures <model.onnx>
Escreve em play/src/core/__fixtures__/net_parity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twenty48.board import GameState  # noqa: E402
from twenty48.encode import encode  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "play/src/core/__fixtures__/net_parity.json"


def random_board(rng: np.random.Generator, n: int, max_exp: int, fill: float) -> GameState:
    cells = tuple(
        int(1 + rng.integers(0, max_exp)) if rng.random() < fill else 0 for _ in range(n * n)
    )
    return GameState(size=n, cells=cells, score=int(rng.integers(0, 30000)))


def main() -> None:
    model_path = sys.argv[1]
    sess = ort.InferenceSession(model_path)
    rng = np.random.default_rng(4040)

    cases = []
    for n in (4, 5, 6):
        for _ in range(12):
            fill = 0.3 + 0.6 * rng.random()
            state = random_board(rng, n, 8, fill)
            x = encode(state)  # (20, n, n) float32
            logits, value = sess.run(None, {"board": x[None, ...]})
            cases.append(
                {
                    "size": n,
                    "cells": list(state.cells),
                    "encoding": x.reshape(-1).astype(float).tolist(),
                    "policy_logits": logits[0].astype(float).tolist(),
                    "value": float(value[0]),
                }
            )
    # Alguns boards extremos (expoente > 17 satura; board cheio).
    big = GameState(4, tuple([18, 1, 2, 3] + [4] * 12), 12345)
    x = encode(big)
    logits, value = sess.run(None, {"board": x[None, ...]})
    cases.append(
        {
            "size": 4,
            "cells": list(big.cells),
            "encoding": x.reshape(-1).astype(float).tolist(),
            "policy_logits": logits[0].astype(float).tolist(),
            "value": float(value[0]),
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": Path(model_path).name, "cases": cases}))
    print(f"wrote {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()

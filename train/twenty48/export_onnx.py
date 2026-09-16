"""Exporta a rede treinada para ONNX — o insumo do browser (onnxruntime-web).

Move para CPU antes de exportar (o device de treino é ortogonal ao artefato: o
.onnx sai idêntico ao de um treino em CUDA). Eixos dinâmicos em batch/altura/
largura preservam o agnosticismo de tamanho no browser.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from .net import Net

# play/public/model.onnx (o asset servido pela plataforma web).
WEB_MODEL = Path(__file__).resolve().parents[2] / "play" / "public" / "model.onnx"


def export_onnx(net: Net, path: str, example_size: int = 4, opset: int = 18) -> None:
    net = net.to("cpu").eval()
    dummy = torch.zeros(1, 20, example_size, example_size, dtype=torch.float32)
    torch.onnx.export(
        net,
        dummy,
        path,
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board": {0: "batch", 2: "height", 3: "width"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=opset,
        # Arquivo único (pesos embutidos) — sem sidecar .onnx.data, mais simples
        # de servir no browser.
        external_data=False,
    )


def export_value_norm(data: dict, out_path: Path) -> dict | None:
    """Grava value_norm.json (μ,σ por tamanho) ao lado do modelo. O MCTS do
    browser precisa desses stats p/ o backup reward-to-go (denorm/renorm). Sem
    normalizador salvo (modelo legado) → não grava."""
    import json

    from .value_norm import ValueNormalizer

    ns = data.get("normalizer")
    if not ns:
        return None
    nz = ValueNormalizer()
    nz.load_state_dict(ns)
    sizes = {}
    for size in sorted(nz._mean):
        mu, sigma = nz._mu_sigma(size)
        sizes[str(size)] = {"mu": mu, "sigma": sigma}
    payload = {"minStd": nz.min_std, "sizes": sizes}
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def export_checkpoint(ckpt: str, out: str) -> None:
    """Carrega um checkpoint (.pt ou pasta de run → best.pt) e exporta para ONNX,
    reconstruindo a arquitetura a partir da config salva. Verifica com ORT e grava
    value_norm.json (μ,σ do reward-to-go) ao lado."""
    p = Path(ckpt)
    if p.is_dir():
        p = p / "best.pt"
    data = torch.load(p, map_location="cpu", weights_only=False)
    cfg = data.get("cfg", {})
    net = Net(cfg.get("channels", 64), cfg.get("blocks", 4))
    net.load_state_dict(data["net"])
    export_onnx(net, out)
    _verify(net, out)
    vn = export_value_norm(data, Path(out).with_name("value_norm.json"))
    vn_msg = f" + value_norm.json ({vn['sizes']})" if vn else " (sem normalizador)"
    print(f"exportado: {p} (iter {data.get('iter')}, eval {data.get('eval_mean_score')}) -> {out}{vn_msg}")


def _verify(net: Net, onnx_path: str) -> None:
    """Confere que o ONNX bate com o PyTorch (ORT), em 4×4 e 5×5."""
    import numpy as np
    import onnxruntime as ort

    from .board import GameState
    from .encode import encode_batch

    net.eval()
    sess = ort.InferenceSession(onnx_path)
    rng = np.random.default_rng(0)
    for n in (4, 5):
        states = [
            GameState(n, tuple(int(rng.integers(0, 7)) for _ in range(n * n)), 0) for _ in range(3)
        ]
        x = encode_batch(states)
        with torch.no_grad():
            tl, tv = net(torch.from_numpy(x))
        ol, ov = sess.run(None, {"board": x})
        ok = np.allclose(tl.numpy(), ol, atol=1e-4) and np.allclose(tv.numpy(), ov, atol=1e-4)
        print(f"  verificação ORT==PyTorch n={n}: {'ok' if ok else 'FALHOU'}")
        if not ok:
            raise SystemExit("paridade ORT falhou — não sirva este modelo")


def _main() -> None:
    ap = argparse.ArgumentParser(description="Exporta um checkpoint para ONNX (e opcionalmente serve).")
    ap.add_argument("--checkpoint", required=True, help="pasta de run (usa best.pt) ou caminho .pt")
    ap.add_argument("--out", help="saída .onnx (padrão: <checkpoint>/model.onnx)")
    ap.add_argument("--to-web", action="store_true", help=f"também copia para {WEB_MODEL}")
    args = ap.parse_args()

    p = Path(args.checkpoint)
    out = args.out or str((p if p.is_dir() else p.parent) / "model.onnx")
    export_checkpoint(args.checkpoint, out)
    if args.to_web:
        WEB_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, WEB_MODEL)
        vn_src = Path(out).with_name("value_norm.json")
        if vn_src.exists():
            shutil.copyfile(vn_src, WEB_MODEL.parent / "value_norm.json")
            print(f"servido: {vn_src} -> {WEB_MODEL.parent / 'value_norm.json'}")
        print(f"servido: {out} -> {WEB_MODEL}")


if __name__ == "__main__":
    _main()

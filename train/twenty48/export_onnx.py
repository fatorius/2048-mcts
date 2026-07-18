"""Exporta a rede treinada para ONNX — o insumo da Fase 4 (onnxruntime-web).

Move para CPU antes de exportar (o device de treino é ortogonal ao artefato: o
.onnx sai idêntico ao de um treino em CUDA). Eixos dinâmicos em batch/altura/
largura preservam o agnosticismo de tamanho no browser.
"""

from __future__ import annotations

import torch

from .net import Net


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
        # de servir no browser na Fase 4.
        external_data=False,
    )

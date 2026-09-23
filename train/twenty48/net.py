"""Rede de duas cabeças, agnóstica ao tamanho (uma rede, tronco compartilhado).

Tronco convolucional (Conv3×3-BN-ReLU) → pooling global (colapsa o n×n variável
num vetor fixo) → duas cabeças densas (valor, política). O pooling é o que torna a
rede agnóstica a n: as cabeças sempre recebem dimensão fixa.

Opções de arquitetura (flags, todas com default = arquitetura ANTIGA p/
compatibilidade de checkpoints; runs novos ligam as melhorias):
  - coord:         canais de coordenada (CoordConv) no input → o tronco enxerga
                   POSIÇÃO (canto/serpente), que o GAP puro apagava. Fica dentro
                   do forward: a entrada externa continua 20 canais.
  - pool="avgmax": concatena average+max pooling global → retém "presença" de uma
                   feature em qualquer célula, não só a média.
  - policy_hidden: camada oculta na cabeça de política (antes era Linear(C→4) cru).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import NUM_CHANNELS

NUM_ACTIONS = 4


class Net(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        blocks: int = 4,
        in_channels: int = NUM_CHANNELS,
        coord: bool = False,
        pool: str = "avg",
        policy_hidden: bool = False,
    ):
        super().__init__()
        assert pool in ("avg", "avgmax"), pool
        self.coord = coord
        self.pool = pool
        trunk_in = in_channels + (2 if coord else 0)  # +2 canais de coordenada

        layers: list[nn.Module] = [
            nn.Conv2d(trunk_in, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks - 1):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
        self.trunk = nn.Sequential(*layers)

        feat = channels * (2 if pool == "avgmax" else 1)  # dim após o pooling
        # Valor: pool → Dense → 1 (sigmoid → [0,1], mesmo referencial do rollout).
        self.value_head = nn.Sequential(
            nn.Linear(feat, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )
        # Política: opcionalmente com camada oculta (as 4 ações não escalam com o board).
        if policy_hidden:
            self.policy_head = nn.Sequential(
                nn.Linear(feat, channels),
                nn.ReLU(inplace=True),
                nn.Linear(channels, NUM_ACTIONS),
            )
        else:
            self.policy_head = nn.Linear(feat, NUM_ACTIONS)

    def _add_coords(self, x: torch.Tensor) -> torch.Tensor:
        # Coordenadas normalizadas em [-1,1] construídas via cumsum de uns: assim a
        # dimensão espacial (dinâmica) flui simbolicamente e o export ONNX preserva
        # os eixos H/W dinâmicos (linspace(steps=h) fixava h como constante). Valor
        # idêntico a linspace(-1,1,·): (i-1)/(dim-1)*2-1.
        b = x.shape[0]
        ones = x.new_ones((b, 1, x.shape[2], x.shape[3]))
        row = ones.cumsum(2)  # 1..h descendo as linhas
        col = ones.cumsum(3)  # 1..w ao longo das colunas
        row = (row - 1) / (row.amax(dim=2, keepdim=True) - 1).clamp_min(1e-6) * 2 - 1
        col = (col - 1) / (col.amax(dim=3, keepdim=True) - 1).clamp_min(1e-6) * 2 - 1
        return torch.cat([x, row, col], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 20, n, n) → (policy_logits (B,4), value (B,) em [0,1])."""
        if self.coord:
            x = self._add_coords(x)
        h = self.trunk(x)
        g = F.adaptive_avg_pool2d(h, 1).flatten(1)  # (B, C) — dimensão fixa
        if self.pool == "avgmax":
            mx = F.adaptive_max_pool2d(h, 1).flatten(1)
            g = torch.cat([g, mx], dim=1)  # (B, 2C)
        value = torch.sigmoid(self.value_head(g)).squeeze(1)
        policy_logits = self.policy_head(g)
        return policy_logits, value


def param_count(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())

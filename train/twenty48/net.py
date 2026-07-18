"""Rede de duas cabeças, agnóstica ao tamanho (uma rede, tronco compartilhado).

Tronco convolucional (Conv3×3-BN-ReLU) → Global Average Pooling (colapsa o n×n
variável num vetor fixo = nº de canais) → duas cabeças densas (valor, política).
O GAP é o que torna a rede agnóstica a n: as cabeças sempre recebem dimensão
fixa. Detalhes no plano §"A rede".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import NUM_CHANNELS

NUM_ACTIONS = 4


class Net(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4, in_channels: int = NUM_CHANNELS):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
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

        # Valor: GAP → Dense → 1 (sigmoid → [0,1], mesmo referencial do rollout).
        self.value_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )
        # Política: Dense → 4 (as 4 ações não escalam com o board).
        self.policy_head = nn.Linear(channels, NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 20, n, n) → (policy_logits (B,4), value (B,) em [0,1])."""
        h = self.trunk(x)
        g = F.adaptive_avg_pool2d(h, 1).flatten(1)  # (B, C) — dimensão fixa
        value = torch.sigmoid(self.value_head(g)).squeeze(1)
        policy_logits = self.policy_head(g)
        return policy_logits, value


def param_count(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())

"""Normalização adaptativa do alvo de valor.

Problema (visto no primeiro run): os resultados de partidas se concentram numa
faixa estreita; com `normalize_score = log2/16` isso vira ~0.05 de spread, então
o value head só prevê a média (perda ~0) e o MCTS perde o sinal de valor.

Correção: padronizar o score por tamanho — z = (score - μ)/σ com μ,σ por média
móvel exponencial (acompanha a melhora do jogo) — e passar por sigmoide → alvo
bem espalhado em [0,1], adaptativo, e ainda dentro do contrato de valor [0,1].
Por-tamanho porque scores de 6×6 >> 4×4.

MODO ADVANTAGE (opcional, `fit_baseline`): o alvo cru reward-to-go (rtg =
score_final − score_atual) é dominado pelo PROGRESSO do jogo (enorme na abertura,
→0 no fim), não pela QUALIDADE do tabuleiro. Padronizar tudo junto faz o eixo
dominante do alvo ser "quanto falta", não "quão bom está" → a rede aprende o
componente trivial e a qualidade-de-tabuleiro vira resíduo ruidoso (v-loss
minúscula, mas inútil p/ discriminar). Correção: subtrair uma BASELINE b(score) =
E[rtg | score] (rtg típico no mesmo estágio) e padronizar o ADVANTAGE a = rtg −
b(score) → o alvo passa a variar com qualidade-de-tabuleiro no estágio fixo. O
backup reward-to-go continua exato porque denorm/renorm readicionam b(score) do
nó (ValueTransform ciente de score)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-6


@dataclass(frozen=True)
class ValueTransform:
    """Converte entre valor normalizado da rede (sigmoide, [0,1]) e reward-to-go
    BRUTO (mesma unidade de score), para o backup reward-to-go do MCTS somar as
    recompensas `gained` das arestas. `ready` = há spread suficiente (σ≥min_std);
    quando falso, o MCTS cai no backup antigo (só valor da folha).

    Com baseline (modo advantage), denorm/renorm recebem o `score` do nó e
    readicionam/subtraem b(score) — mantendo o backup exato mesmo com o alvo
    centrado no advantage. Sem baseline (knots vazio), b(score)=0 e o
    comportamento é idêntico ao legado (padronização pura do rtg)."""

    mu: float
    sigma: float
    ready: bool
    knots: tuple = ()  # scores-nó da baseline (crescentes); vazio = modo legado
    base: tuple = ()   # rtg típico (baseline) em cada nó

    def _baseline(self, score: float) -> float:
        if not self.knots:
            return 0.0
        return float(np.interp(score, self.knots, self.base))

    def denorm(self, v: float, score: float = 0.0) -> float:
        """valor normalizado [0,1] → reward-to-go bruto (b(score) + logit·σ + μ)."""
        c = min(1.0 - _EPS, max(_EPS, v))
        return self._baseline(score) + self.mu + self.sigma * math.log(c / (1.0 - c))

    def renorm(self, raw: float, score: float = 0.0) -> float:
        """reward-to-go bruto → valor normalizado [0,1] (sigmoide do advantage)."""
        z = (raw - self._baseline(score) - self.mu) / self.sigma
        if z <= -60.0:
            return 0.0
        if z >= 60.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))


def _fit_knots(scores: np.ndarray, rtg: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Baseline não-paramétrica b(score)=E[rtg|score]: bins por quantil de score,
    (score médio, rtg médio) por bin, garantindo nós estritamente crescentes."""
    qs = np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1))
    knots: list[float] = []
    base: list[float] = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        m = (scores >= lo) & (scores <= hi) if i == n_bins - 1 else (scores >= lo) & (scores < hi)
        if not m.any():
            continue
        k = float(scores[m].mean())
        if knots and k <= knots[-1]:  # mantém estritamente crescente p/ np.interp
            continue
        knots.append(k)
        base.append(float(rtg[m].mean()))
    return np.array(knots, dtype=np.float64), np.array(base, dtype=np.float64)


class ValueNormalizer:
    def __init__(self, momentum: float = 0.02, min_std: float = 1.0):
        self.momentum = momentum
        self.min_std = min_std
        self._mean: dict[int, float] = {}
        self._sq: dict[int, float] = {}  # EMA de x^2
        self._count: dict[int, int] = {}
        # Modo advantage (preenchido por fit_baseline; ausente = modo legado).
        self._bl_knots: dict[int, np.ndarray] = {}
        self._bl_base: dict[int, np.ndarray] = {}
        self._adv_mean: dict[int, float] = {}
        self._adv_sq: dict[int, float] = {}

    def update(self, size: int, score: float) -> None:
        x = float(score)
        a = self.momentum
        if size not in self._mean:
            self._mean[size] = x
            self._sq[size] = x * x
            self._count[size] = 1
        else:
            self._mean[size] = (1 - a) * self._mean[size] + a * x
            self._sq[size] = (1 - a) * self._sq[size] + a * x * x
            self._count[size] += 1

    def recalibrate(self, size: int, values) -> None:
        """Define μ,σ a partir da distribuição REAL de reward-to-go (média/var da
        amostra), SEM viés de ordem. Corrige o EMA por-posição (janela ~50 << jogo
        ~1000), que seguia a trajetória decrescente do rtg dentro do jogo e saturava
        o valor (μ,σ ~10× pequenos demais). Ver histórico."""
        v = np.asarray(values, dtype=np.float64)
        if v.size == 0:
            return
        self._mean[size] = float(v.mean())
        self._sq[size] = float((v * v).mean())
        self._count[size] = int(v.size)

    def fit_baseline(self, size: int, scores, rtg, n_bins: int = 32) -> None:
        """Ajusta a baseline b(score)=E[rtg|score] e as estatísticas do advantage
        a=rtg−b(score) a partir da distribuição real do buffer (modo advantage)."""
        s = np.asarray(scores, dtype=np.float64)
        r = np.asarray(rtg, dtype=np.float64)
        if s.size == 0:
            return
        n_bins = max(2, min(n_bins, s.size // 8))
        knots, base = _fit_knots(s, r, n_bins)
        if knots.size < 2:  # fallback: baseline achatada na média
            knots = np.array([s.min(), s.max() + 1.0])
            base = np.array([r.mean(), r.mean()])
        adv = r - np.interp(s, knots, base)
        self._bl_knots[size] = knots
        self._bl_base[size] = base
        self._adv_mean[size] = float(adv.mean())
        self._adv_sq[size] = float((adv * adv).mean())

    def _mu_sigma(self, size: int) -> tuple[float, float]:
        if size not in self._mean:
            return 0.0, 0.0
        mu = self._mean[size]
        var = max(self._sq[size] - mu * mu, 0.0)
        return mu, math.sqrt(var)

    def _adv_mu_sigma(self, size: int) -> tuple[float, float]:
        mu = self._adv_mean[size]
        var = max(self._adv_sq[size] - mu * mu, 0.0)
        return mu, math.sqrt(var)

    def has_baseline(self, size: int) -> bool:
        return size in self._bl_knots

    def normalize(self, score: float, size: int) -> float:
        """score bruto → [0,1] padronizado. 0.5 enquanto não há spread suficiente."""
        mu, sigma = self._mu_sigma(size)
        if sigma < self.min_std:
            return 0.5
        z = (float(score) - mu) / sigma
        return 1.0 / (1.0 + math.exp(-z))

    def normalize_array(self, scores, size: int) -> np.ndarray:
        mu, sigma = self._mu_sigma(size)
        if sigma < self.min_std:
            return np.full(len(scores), 0.5, dtype=np.float32)
        z = (np.asarray(scores, dtype=np.float64) - mu) / sigma
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def advantage_array(self, rtg, scores, size: int) -> np.ndarray:
        """alvo em modo advantage: sigmoide do advantage padronizado a=rtg−b(score).
        Cai em normalize_array (rtg puro) se a baseline ainda não foi ajustada."""
        if size not in self._bl_knots:
            return self.normalize_array(rtg, size)
        mu, sigma = self._adv_mu_sigma(size)
        if sigma < self.min_std:
            return np.full(len(rtg), 0.5, dtype=np.float32)
        b = np.interp(np.asarray(scores, dtype=np.float64), self._bl_knots[size], self._bl_base[size])
        adv = np.asarray(rtg, dtype=np.float64) - b
        z = (adv - mu) / sigma
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def transform(self, size: int) -> "ValueTransform":
        """Adaptador por-tamanho para o backup reward-to-go do MCTS: converte
        valor normalizado [0,1] ↔ reward-to-go bruto, com os μ,σ correntes. Em modo
        advantage (baseline ajustada), carrega b(score) e usa μ_A,σ_A."""
        if size in self._bl_knots:
            mu, sigma = self._adv_mu_sigma(size)
            return ValueTransform(
                mu, sigma, sigma >= self.min_std,
                knots=tuple(self._bl_knots[size].tolist()),
                base=tuple(self._bl_base[size].tolist()),
            )
        mu, sigma = self._mu_sigma(size)
        return ValueTransform(mu, sigma, sigma >= self.min_std)

    def terminal_value_fn(self):
        """Fn para o MCTS avaliar folhas terminais na MESMA escala do alvo."""
        return lambda state: self.normalize(state.score, state.size)

    def state_dict(self) -> dict:
        return {
            "mean": dict(self._mean),
            "sq": dict(self._sq),
            "count": dict(self._count),
            "bl_knots": {k: v.tolist() for k, v in self._bl_knots.items()},
            "bl_base": {k: v.tolist() for k, v in self._bl_base.items()},
            "adv_mean": dict(self._adv_mean),
            "adv_sq": dict(self._adv_sq),
        }

    def load_state_dict(self, state: dict) -> None:
        # As chaves de tamanho podem virar str após serialização; normaliza p/ int.
        self._mean = {int(k): v for k, v in state.get("mean", {}).items()}
        self._sq = {int(k): v for k, v in state.get("sq", {}).items()}
        self._count = {int(k): v for k, v in state.get("count", {}).items()}
        self._bl_knots = {int(k): np.asarray(v, dtype=np.float64) for k, v in state.get("bl_knots", {}).items()}
        self._bl_base = {int(k): np.asarray(v, dtype=np.float64) for k, v in state.get("bl_base", {}).items()}
        self._adv_mean = {int(k): v for k, v in state.get("adv_mean", {}).items()}
        self._adv_sq = {int(k): v for k, v in state.get("adv_sq", {}).items()}

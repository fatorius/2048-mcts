# Fase 3 — Treino: self-play + RL (estilo AlphaZero)

Loop de RL offline em Python/PyTorch (MPS local) que destila a busca numa rede
barata, e exporta o modelo para ONNX (insumo da Fase 4). O motor e o MCTS são um
**espelho fiel** do `core/` TS — verificado por testes de paridade.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch onnx onnxscript onnxruntime numpy pytest
```

Python **3.12** (mais maduro para MPS que o 3.14); PyTorch 2.13 com MPS.

## Uso

```bash
.venv/bin/python -m twenty48.train --smoke        # valida o pipeline (rápido)
.venv/bin/python -m twenty48.train --iterations 40 --games-per-iter 24 --sims 100
.venv/bin/python -m twenty48.train --sizes 4 5 6  # treino multi-tamanho
.venv/bin/python -m pytest                         # testes (paridade + unidade)
```

Saídas em `checkpoints/`: `latest.pt`, `best.pt` (por score de eval), `model.onnx`.

## Arquitetura do loop

Um processo, um laço (`train.py`): a cada iteração —
1. **Self-play** (`self_play.py`): a rede atual joga contra si mesma com MCTS
   guiado por ela; grava (board, distribuição de visitas, resultado) por posição
   no **replay buffer** (`buffer.py`, janela FIFO por tamanho).
2. **Treino** (`train.py`): passos de gradiente amostrando o buffer; perda
   `MSE(valor, z) + CE(política, visitas) + L2`.
3. **Eval + checkpoint** (`evaluate.py`): partidas gulosas → score médio, taxa
   2048/4096 (o sinal de progresso — não há oponente).

**Warm start:** a iteração 0 gera dados com o MCTS-rollout da Fase 1 (a rede
recém-inicializada tem value head de ruído); a partir da iteração 1, self-play
guiado pela rede.

## Módulos

| Arquivo | Papel |
|---|---|
| `twenty48/board.py` | Motor 2048 — espelho de `board.ts` (paridade testada). |
| `twenty48/encode.py` | Codificação 20×n×n (17 one-hot + 3 auxiliares). |
| `twenty48/net.py` | Rede de 2 cabeças, agnóstica a `n` (tronco conv → GAP → value/policy). |
| `twenty48/mcts.py` | MCTS com chance nodes + avaliação de folhas EM LOTE (virtual loss). |
| `twenty48/evaluators.py` | Fronteira `evaluate`: uniform / rollout / rede. |
| `twenty48/self_play.py` | Gera uma partida e grava posições. |
| `twenty48/buffer.py` | Replay buffer por tamanho. |
| `twenty48/train.py` | Orquestrador do loop. |
| `twenty48/evaluate.py` | Métricas de progresso. |
| `twenty48/export_onnx.py` | Export para ONNX (CPU, eixos dinâmicos). |

## Paridade com o TS

`tests/test_parity.py` verifica o motor Python contra fixtures geradas pelo lado
TS (`cd play && npm run gen:fixtures`). Blinda contra divergência silenciosa entre
a busca de treino (Python) e a que roda no browser (TS) — 187 casos exatos:
movimentos, terminal, ações legais, distribuição de spawn, normalização.

## Notas de MPS

- Device escolhido automaticamente (`mps` > `cuda` > `cpu`). Batch de folhas do
  MCTS torna a inferência barata: ~0.5 ms para 32–64 posições em lote (vs. CPU).
- Treino em float32 (estável em MPS; o `.onnx` exportado é float32 de qualquer
  forma — o device de treino é ortogonal ao artefato).

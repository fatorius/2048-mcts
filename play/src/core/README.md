# Núcleo (Fase 1) — motor 2048 + MCTS

Módulo framework-agnóstico (sem React/DOM). É a peça reusada sem porte pela
plataforma (Fase 2) e pelo deploy no cliente (Fase 4). Ver `plano_2048_mcts_rl.md`.

## Contratos congelados

Fixados **antes** do código, compartilhados entre a busca (JS/TS), a codificação
de entrada da rede (Python, Fase 3) e a serialização entre runtimes. Definidos em
[`types.ts`](./types.ts). Não alterar sem migrar todas as fases.

1. **Estado do tabuleiro** — `GameState { size, cells, score }`. `cells` é um
   `Uint8Array` row-major de **expoentes** (`0` = vazio, `k` = tile `2^k`).
   Expoente e não valor: casa direto com o one-hot por expoente da entrada da
   rede e serializa como bytes `0..17`.
2. **`evaluate(state) → { policy, value }`** (`Evaluator`) — fronteira neutra que
   o MCTS chama nas folhas. `policy`: prior sobre as 4 ações (soma 1). `value`:
   qualidade em `[0,1]`, maior = melhor.
   - Fase 1: prior uniforme + valor por rollout aleatório ([`evaluate.ts`](./evaluate.ts)).
   - Fase 3/4: a rede substitui o stub **sem tocar o núcleo da busca**.

Ações (ordem canônica, igual à cabeça de política): `0=Up, 1=Right, 2=Down, 3=Left`.

## Arquivos

| Arquivo | Papel |
|---|---|
| `types.ts` | Contratos congelados: `GameState`, `Evaluator`, `Action`. |
| `rng.ts` | PRNG seedável (mulberry32) — reprodutibilidade p/ testes de paridade. |
| `board.ts` | Motor: `applyMove` (puro), spawn, terminal, `spawnOutcomes` (distribuição exata). |
| `evaluate.ts` | Stub de `evaluate`: rollout aleatório + `normalizeScore`. |
| `mcts.ts` | `runMcts` — busca com chance nodes explícitos e backup ponderado. |
| `agent.ts` | `playGame` — encadeia buscas até o fim; `onStep` p/ visualização (Fase 2). |

## MCTS com chance nodes

Alterna nós de **decisão** (4 ações, seleção PUCT) e nós de **chance** (spawn).
Os resultados de spawn são amostrados na descida ∝ sua probabilidade real
(90% "2", 10% "4", uniforme sobre vazias), então a média de visitas de um chance
node é um estimador não-viesado da expectativa exata `Σ pᵢ·V(i)` — é o "backup
ponderado pela distribuição exata" do plano, realizado via frequência de
amostragem. Como 2048 é solo contra o acaso, o valor da folha é somado em todas
as arestas do caminho, sem alternância de sinal.

## Uso

```ts
import { initialState, runMcts, makeRandomRolloutEvaluator, mulberry32 } from './core';

const rng = mulberry32(1);
const state = initialState(4, rng);
const result = runMcts(state, {
  simulations: 300,
  cPuct: 1.5,
  evaluator: makeRandomRolloutEvaluator(rng),
  rng,
});
// result.visits -> contagem por ação (a "política do MCTS"); result.bestAction
```

## Scripts

- `npm test` — suíte Vitest (motor + invariantes da busca).
- `npm run bench -- [games] [sims] [size] [cPuct] [seed]` — critério de saída:
  fração de partidas 4×4 que alcançam 2048 com rollouts aleatórios.

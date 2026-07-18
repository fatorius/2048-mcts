// Perfil de latência POR LANCE (não por partida) e decomposição do custo.
//
// Mede ms/lance variando as simulações, com dois avaliadores:
//   - rollout : stub da Fase 1 (rollout aleatório até o fim) — CARO.
//   - const   : avaliador O(1) (política uniforme + valor fixo). Aproxima a
//               ESTRUTURA de custo de um eval de rede (busca + mecânica de board,
//               SEM o forward pass). A diferença rollout-const = custo do rollout,
//               que a rede ELIMINA no produto final.
//
// Uso: npm run profile

import { mulberry32, type RNG } from '../src/core/rng';
import { makeRandomRolloutEvaluator } from '../src/core/evaluate';
import { runMcts } from '../src/core/mcts';
import { initialState, isTerminal, step } from '../src/core/board';
import type { Evaluator } from '../src/core/types';

const HORIZON = 120; // lances medidos por corrida
const SIM_COUNTS = [25, 50, 100, 200, 400, 800];
const SEEDS = [1, 2];

const constEval: Evaluator = () => ({ policy: [0.25, 0.25, 0.25, 0.25], value: 0.5 });

function msPerMove(simulations: number, makeEval: (rng: RNG) => Evaluator): number {
  let totalMs = 0;
  let totalMoves = 0;
  for (const seed of SEEDS) {
    const rng = mulberry32(seed);
    const evaluator = makeEval(rng);
    let state = initialState(4, rng);
    let moves = 0;
    while (!isTerminal(state) && moves < HORIZON) {
      const t0 = performance.now();
      const res = runMcts(state, { simulations, cPuct: 1.5, evaluator, rng });
      totalMs += performance.now() - t0;
      if (res.bestAction === -1) break;
      state = step(state, res.bestAction, rng).state;
      moves++;
    }
    totalMoves += moves;
  }
  return totalMs / totalMoves;
}

console.log('ms por LANCE @ 4x4 (média sobre 2 seeds, horizonte 120 lances)\n');
console.log('  sims | rollout(F1) |  const(~rede) | custo rollout');
console.log('  -----+-------------+---------------+--------------');
for (const sims of SIM_COUNTS) {
  const rollout = msPerMove(sims, (rng) => makeRandomRolloutEvaluator(rng));
  const cst = msPerMove(sims, () => constEval);
  console.log(
    `  ${String(sims).padStart(4)} | ${rollout.toFixed(2).padStart(9)}ms | ` +
      `${cst.toFixed(2).padStart(11)}ms | ${(rollout - cst).toFixed(2).padStart(8)}ms ` +
      `(${((100 * (rollout - cst)) / rollout).toFixed(0)}%)`,
  );
}

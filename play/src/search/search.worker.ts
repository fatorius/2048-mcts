// Search worker — roda o MCTS fora da main thread para manter a UI fluida
// (e é o lugar natural para a inferência WebGPU da Fase 4).
//
// Mantém um RNG persistente entre buscas: reamostra spawns/rollouts com um fluxo
// contínuo, resetado por 'init'. O núcleo (core/) é reusado sem alteração.

import { type Evaluator, makeRandomRolloutEvaluator, mulberry32, runMcts, type RNG } from '../core';
import type { GameState } from '../core';
import type { WorkerRequest, WorkerResponse } from './protocol';

let rng: RNG = mulberry32(1);
let evaluator: Evaluator = makeRandomRolloutEvaluator(rng);

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;

  if (msg.type === 'init') {
    rng = mulberry32(msg.seed);
    evaluator = makeRandomRolloutEvaluator(rng);
    post({ type: 'ready' });
    return;
  }

  // type === 'search'
  const state: GameState = { size: msg.size, cells: msg.cells, score: msg.score };
  const t0 = performance.now();
  const result = runMcts(state, {
    simulations: msg.simulations,
    cPuct: msg.cPuct,
    evaluator,
    rng,
  });
  post({ type: 'result', id: msg.id, result, elapsedMs: performance.now() - t0 });
};

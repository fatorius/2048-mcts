// Search worker — roda o MCTS fora da main thread para manter a UI fluida.
// Fase 4: além do stub de rollout (síncrono), carrega a rede ONNX e usa a busca
// ASSÍNCRONA (runMctsAsync) quando o modo é 'net'. O núcleo é reusado sem mudança.

import {
  type Evaluator,
  makeRandomRolloutEvaluator,
  mulberry32,
  runMcts,
  runMctsAsync,
  type RNG,
} from '../core';
import type { GameState } from '../core';
import { NetEvaluator } from './netEvaluator';
import type { WorkerRequest, WorkerResponse } from './protocol';

let rng: RNG = mulberry32(1);
let rolloutEval: Evaluator = makeRandomRolloutEvaluator(rng);
let net: NetEvaluator | null = null;

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;

  if (msg.type === 'init') {
    rng = mulberry32(msg.seed);
    rolloutEval = makeRandomRolloutEvaluator(rng);
    post({ type: 'ready' });
    return;
  }

  if (msg.type === 'loadNet') {
    try {
      net = new NetEvaluator();
      const info = await net.load(msg.modelUrl);
      post({ type: 'netLoaded', ok: true, backend: info.backend });
    } catch (err) {
      net = null;
      post({ type: 'netLoaded', ok: false, error: String(err) });
    }
    return;
  }

  // type === 'search'
  const state: GameState = { size: msg.size, cells: msg.cells, score: msg.score };
  const t0 = performance.now();
  const result =
    msg.mode === 'net' && net
      ? await runMctsAsync(state, {
          simulations: msg.simulations,
          cPuct: msg.cPuct,
          evaluator: net.evaluate,
          rng,
        })
      : runMcts(state, { simulations: msg.simulations, cPuct: msg.cPuct, evaluator: rolloutEval, rng });
  post({ type: 'result', id: msg.id, result, elapsedMs: performance.now() - t0 });
};

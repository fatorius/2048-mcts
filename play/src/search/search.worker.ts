// Search worker — roda o MCTS fora da main thread para manter a UI fluida.
// Além do stub de rollout (síncrono), carrega a rede ONNX e usa a busca
// ASSÍNCRONA (runMctsAsync) quando o modo é 'net'. O núcleo é reusado sem mudança.

import {
  type Evaluator,
  makeRandomRolloutEvaluator,
  makeValueTransform,
  mulberry32,
  runMcts,
  runMctsAsync,
  type RNG,
  type ValueTransform,
} from '../core';
import type { GameState } from '../core';
import { NetEvaluator } from './netEvaluator';
import type { WorkerRequest, WorkerResponse } from './protocol';

let rng: RNG = mulberry32(1);
let rolloutEval: Evaluator = makeRandomRolloutEvaluator(rng);
let net: NetEvaluator | null = null;
// Transforms reward-to-go por tamanho (μ,σ do treino). null = modelo sem
// value_norm.json (legado) → busca cai no backup antigo (score-total).
let valueNorm: Record<number, ValueTransform> | null = null;

/** Carrega os μ,σ do reward-to-go de value_norm.json (mesmo diretório do modelo).
 *  Ausente/erro → null (compatível com modelos antigos). */
async function loadValueNorm(modelUrl: string): Promise<Record<number, ValueTransform> | null> {
  const url = modelUrl.replace(/[^/]*$/, 'value_norm.json');
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = (await res.json()) as {
      minStd?: number;
      sizes: Record<string, { mu: number; sigma: number }>;
    };
    const minStd = data.minStd ?? 1;
    const out: Record<number, ValueTransform> = {};
    for (const [size, s] of Object.entries(data.sizes)) {
      out[Number(size)] = makeValueTransform(s.mu, s.sigma, minStd);
    }
    return out;
  } catch {
    return null;
  }
}

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
      valueNorm = await loadValueNorm(msg.modelUrl);
      post({ type: 'netLoaded', ok: true, backend: info.backend });
    } catch (err) {
      net = null;
      valueNorm = null;
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
          valueTransform: valueNorm?.[msg.size],
        })
      : runMcts(state, { simulations: msg.simulations, cPuct: msg.cPuct, evaluator: rolloutEval, rng });
  post({ type: 'result', id: msg.id, result, elapsedMs: performance.now() - t0 });
};

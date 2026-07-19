// NetEvaluator — a rede treinada atrás da interface `evaluate`, via onnxruntime-web.
// É o coração da Fase 4: o MCTS (assíncrono) passa a chamar a rede no lugar do
// stub de rollout, sem que o núcleo da busca mude.
//
// Inferência assíncrona (ORT sempre retorna Promise) → usada com runMctsAsync.
// Roda no Worker; backend WebGPU de preferência, WASM como fallback.

import * as ort from 'onnxruntime-web';
import type { AsyncEvaluator, Evaluation, GameState } from '../core';
import { ACTIONS } from '../core';
import { applyMove } from '../core/board';
import { encode, NUM_CHANNELS } from '../core/encode';

// Assets do runtime ORT (.wasm/.mjs) servidos por CDN — o `exports` do pacote não
// expõe os subpaths de dist, e o import dinâmico do .mjs a partir de /public quebra
// no dev do Vite. CDN é URL externa: o Vite não intercepta. (Para self-host em
// produção, copiar o dist e servir estático; é detalhe de deploy.) numThreads=1
// evita exigir COOP/COEP (SharedArrayBuffer).
const ORT_VERSION = '1.27.0';
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = 1;

export interface NetInfo {
  readonly backend: string;
}

export class NetEvaluator {
  private session: ort.InferenceSession | null = null;
  backend = 'unknown';

  /** Carrega o modelo. Retorna o backend efetivo (webgpu | wasm). */
  async load(modelUrl: string): Promise<NetInfo> {
    try {
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu', 'wasm'],
      });
      this.backend = 'webgpu';
    } catch {
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
      });
      this.backend = 'wasm';
    }
    return { backend: this.backend };
  }

  /** AsyncEvaluator: state → { policy (softmax mascarado sobre legais), value ∈ [0,1] }. */
  evaluate: AsyncEvaluator = async (state: GameState): Promise<Evaluation> => {
    if (!this.session) throw new Error('NetEvaluator não carregado');
    const n = state.size;
    const input = new ort.Tensor('float32', encode(state), [1, NUM_CHANNELS, n, n]);
    const out = await this.session.run({ board: input });
    const logits = out.policy_logits.data as Float32Array;
    const value = (out.value.data as Float32Array)[0];
    return { policy: maskedSoftmax(logits, state), value };
  };
}

/** Softmax das 4 logits restrito às ações legais (ilegais → 0), igual ao _expand
 *  do treino (twenty48/mcts.py). */
function maskedSoftmax(logits: Float32Array, state: GameState): number[] {
  const legal = ACTIONS.map((a) => applyMove(state, a).moved);
  let max = -Infinity;
  for (const a of ACTIONS) if (legal[a] && logits[a] > max) max = logits[a];
  if (max === -Infinity) return [0, 0, 0, 0]; // terminal (nenhuma legal)
  const exp = [0, 0, 0, 0];
  let sum = 0;
  for (const a of ACTIONS) {
    if (legal[a]) {
      exp[a] = Math.exp(logits[a] - max);
      sum += exp[a];
    }
  }
  return ACTIONS.map((a) => (legal[a] ? exp[a] / sum : 0));
}

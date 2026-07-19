// Wrapper de main thread sobre o search worker, com API baseada em Promise.

import type { GameState, SearchResult } from '../core';
import type { EvaluatorMode, WorkerRequest, WorkerResponse } from './protocol';

export interface SearchOutcome {
  readonly result: SearchResult;
  readonly elapsedMs: number;
}

export interface NetLoadResult {
  readonly ok: boolean;
  readonly backend?: string;
  readonly error?: string;
}

export class SearchClient {
  private worker: Worker;
  private pending = new Map<number, (o: SearchOutcome) => void>();
  private netLoadResolve: ((r: NetLoadResult) => void) | null = null;
  private nextId = 1;

  constructor(seed: number) {
    this.worker = new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'result') {
        const resolve = this.pending.get(msg.id);
        if (resolve) {
          this.pending.delete(msg.id);
          resolve({ result: msg.result, elapsedMs: msg.elapsedMs });
        }
      } else if (msg.type === 'netLoaded') {
        this.netLoadResolve?.({ ok: msg.ok, backend: msg.backend, error: msg.error });
        this.netLoadResolve = null;
      }
    };
    this.post({ type: 'init', seed });
  }

  /** Reinicia o fluxo de RNG do worker (nova partida). */
  reset(seed: number): void {
    this.post({ type: 'init', seed });
  }

  /** Carrega o modelo ONNX no worker (uma vez). Resolve com o backend efetivo. */
  loadNet(modelUrl: string): Promise<NetLoadResult> {
    return new Promise<NetLoadResult>((resolve) => {
      this.netLoadResolve = resolve;
      this.post({ type: 'loadNet', modelUrl });
    });
  }

  search(
    state: GameState,
    simulations: number,
    cPuct: number,
    mode: EvaluatorMode,
  ): Promise<SearchOutcome> {
    const id = this.nextId++;
    return new Promise<SearchOutcome>((resolve) => {
      this.pending.set(id, resolve);
      this.post({
        type: 'search',
        id,
        cells: state.cells,
        size: state.size,
        score: state.score,
        simulations,
        cPuct,
        mode,
      });
    });
  }

  terminate(): void {
    this.worker.terminate();
    this.pending.clear();
  }

  private post(msg: WorkerRequest): void {
    this.worker.postMessage(msg);
  }
}

// Wrapper de main thread sobre o search worker, com API baseada em Promise.

import type { GameState, SearchResult } from '../core';
import type { WorkerRequest, WorkerResponse } from './protocol';

export interface SearchOutcome {
  readonly result: SearchResult;
  readonly elapsedMs: number;
}

export class SearchClient {
  private worker: Worker;
  private pending = new Map<number, (o: SearchOutcome) => void>();
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
      }
    };
    this.post({ type: 'init', seed });
  }

  /** Reinicia o fluxo de RNG do worker (nova partida). */
  reset(seed: number): void {
    this.post({ type: 'init', seed });
  }

  search(state: GameState, simulations: number, cPuct: number): Promise<SearchOutcome> {
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

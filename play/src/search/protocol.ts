// Protocolo de mensagens entre a main thread e o search worker.
// GameState viaja como campos primitivos (cells é Uint8Array, clonável por
// structured clone) — sem transferência, pois a main thread mantém seu estado.

import type { SearchResult } from '../core';

export interface SearchRequest {
  readonly type: 'search';
  readonly id: number;
  readonly cells: Uint8Array;
  readonly size: number;
  readonly score: number;
  readonly simulations: number;
  readonly cPuct: number;
}

export interface InitRequest {
  readonly type: 'init';
  readonly seed: number;
}

export type WorkerRequest = SearchRequest | InitRequest;

export interface ReadyResponse {
  readonly type: 'ready';
}

export interface SearchResponse {
  readonly type: 'result';
  readonly id: number;
  readonly result: SearchResult;
  /** Tempo de parede da busca, em ms (para o painel de diagnóstico). */
  readonly elapsedMs: number;
}

export type WorkerResponse = ReadyResponse | SearchResponse;

// Protocolo de mensagens entre a main thread e o search worker.
// GameState viaja como campos primitivos (cells é Uint8Array, clonável por
// structured clone) — sem transferência, pois a main thread mantém seu estado.

import type { SearchResult } from '../core';

export type EvaluatorMode = 'rollout' | 'net';

export interface SearchRequest {
  readonly type: 'search';
  readonly id: number;
  readonly cells: Uint8Array;
  readonly size: number;
  readonly score: number;
  readonly simulations: number;
  readonly cPuct: number;
  /** rollout (stub Fase 1) ou net (rede ONNX). */
  readonly mode: EvaluatorMode;
}

export interface InitRequest {
  readonly type: 'init';
  readonly seed: number;
}

/** Carrega o modelo ONNX no worker (lazy, ao ligar o modo rede). */
export interface LoadNetRequest {
  readonly type: 'loadNet';
  readonly modelUrl: string;
}

export type WorkerRequest = SearchRequest | InitRequest | LoadNetRequest;

export interface ReadyResponse {
  readonly type: 'ready';
}

export interface NetLoadedResponse {
  readonly type: 'netLoaded';
  readonly ok: boolean;
  readonly backend?: string;
  readonly error?: string;
}

export interface SearchResponse {
  readonly type: 'result';
  readonly id: number;
  readonly result: SearchResult;
  /** Tempo de parede da busca, em ms (para o painel de diagnóstico). */
  readonly elapsedMs: number;
}

export type WorkerResponse = ReadyResponse | NetLoadedResponse | SearchResponse;

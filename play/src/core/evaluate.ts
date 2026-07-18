// ============================================================================
// Stub de `evaluate` da Fase 1 — prior uniforme + valor por rollout aleatório.
//
// Atrás da interface Evaluator (types.ts). Na Fase 3/4 a rede substitui isto
// sem que o MCTS mude. O núcleo da busca nunca embute avaliação.
// ============================================================================

import type { Evaluation, Evaluator, GameState } from './types';
import { applyMove, isTerminal, legalActions, spawnRandom } from './board';
import { type RNG, randInt } from './rng';

/**
 * Escala de normalização do valor. O valor é log2(score+1)/VALUE_SCALE saturado
 * em [0,1]. Com 16, um jogo 4×4 que alcança 2048 (score ~20k) mapeia para ~0.89;
 * headroom até 2^16 mantém o valor < 1. Monotônico no score => preserva a ordem
 * relativa que o MCTS usa para comparar folhas.
 */
export const VALUE_SCALE = 16;

/** Normaliza uma pontuação bruta para o valor em [0,1] do contrato. */
export function normalizeScore(score: number): number {
  const v = Math.log2(score + 1) / VALUE_SCALE;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Prior uniforme sobre as 4 ações (stub da cabeça de política). */
const UNIFORM_POLICY: readonly number[] = [0.25, 0.25, 0.25, 0.25];

/**
 * Joga uma partida aleatória (movimentos legais uniformes + spawns reais) a
 * partir de `state` até o fim, e devolve a pontuação final. É a estimativa de
 * valor de Monte Carlo do stub.
 */
export function randomRollout(state: GameState, rng: RNG): number {
  let cells = state.cells;
  let score = state.score;
  const size = state.size;

  for (;;) {
    const cur: GameState = { size, cells, score };
    if (isTerminal(cur)) return score;
    const legal = legalActions(cur);
    if (legal.length === 0) return score;
    const action = legal[randInt(rng, legal.length)];
    const res = applyMove(cur, action);
    score += res.gained;
    cells = spawnRandom(res.cells, rng);
  }
}

export interface RolloutOptions {
  /** Quantos rollouts promediar por avaliação (mais = menos variância). */
  readonly rollouts?: number;
}

/**
 * Constrói o Evaluator stub da Fase 1: política uniforme + valor = média de N
 * rollouts aleatórios, normalizada para [0,1].
 */
export function makeRandomRolloutEvaluator(rng: RNG, opts: RolloutOptions = {}): Evaluator {
  const rollouts = Math.max(1, opts.rollouts ?? 1);
  return (state: GameState): Evaluation => {
    if (isTerminal(state)) {
      return { policy: UNIFORM_POLICY, value: normalizeScore(state.score) };
    }
    let total = 0;
    for (let i = 0; i < rollouts; i++) total += randomRollout(state, rng);
    return { policy: UNIFORM_POLICY, value: normalizeScore(total / rollouts) };
  };
}

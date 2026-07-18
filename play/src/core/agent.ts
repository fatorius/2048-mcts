// ============================================================================
// Driver de partida — encadeia buscas MCTS até o fim do jogo.
//
// A cada estado: roda o MCTS, joga a ação mais visitada, spawna de verdade.
// A Fase 2 reusa este loop expondo `onStep` para visualizar as visitas por
// jogada; a Fase 3 o reusa para gerar dados de self-play.
// ============================================================================

import type { Evaluator, GameState } from './types';
import { type RNG } from './rng';
import { initialState, isTerminal, maxExponent, step } from './board';
import { runMcts, type SearchResult } from './mcts';

export interface PlayConfig {
  readonly simulations: number;
  readonly cPuct: number;
  /**
   * Fábrica do avaliador. Recebe o RNG da partida para que rollouts/rede e
   * spawns compartilhem a mesma fonte determinística (reprodutível por seed).
   */
  readonly makeEvaluator: (rng: RNG) => Evaluator;
  readonly terminalValue?: (state: GameState) => number;
  /** Callback por jogada (Fase 2: visualização; depuração). */
  readonly onStep?: (state: GameState, result: SearchResult) => void;
  /** Teto de jogadas de segurança (evita loop infinito por bug). */
  readonly maxMoves?: number;
}

export interface GameResult {
  readonly score: number;
  readonly maxExponent: number;
  readonly maxTile: number;
  readonly moves: number;
  readonly finalState: GameState;
}

/** Joga uma partida completa n×n guiada por MCTS. Determinístico por RNG. */
export function playGame(size: number, cfg: PlayConfig, rng: RNG): GameResult {
  const evaluator = cfg.makeEvaluator(rng);
  const maxMoves = cfg.maxMoves ?? Number.POSITIVE_INFINITY;
  let state = initialState(size, rng);
  let moves = 0;

  while (!isTerminal(state) && moves < maxMoves) {
    const result = runMcts(state, {
      simulations: cfg.simulations,
      cPuct: cfg.cPuct,
      evaluator,
      rng,
      terminalValue: cfg.terminalValue,
    });
    if (result.bestAction === -1) break;
    cfg.onStep?.(state, result);
    const next = step(state, result.bestAction, rng);
    if (!next.moved) break; // não deveria ocorrer: bestAction é legal
    state = next.state;
    moves++;
  }

  const exp = maxExponent(state);
  return { score: state.score, maxExponent: exp, maxTile: 1 << exp, moves, finalState: state };
}

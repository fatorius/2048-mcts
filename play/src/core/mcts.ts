// ============================================================================
// MCTS com CHANCE NODES EXPLÍCITOS (estilo AlphaZero, single-player estocástico)
// ----------------------------------------------------------------------------
// Alterna nós de decisão (4 ações, seleção PUCT) e nós de chance (spawn). O
// backup é ponderado pela distribuição EXATA de spawn: como os resultados são
// amostrados na descida ∝ sua probabilidade real, a média de visitas de um
// chance node é um estimador não-viesado da expectativa exata Σ p_i · V(i).
//
// Como 2048 é solo contra o acaso (sem oponente), o valor tem um único
// referencial "quão bom é o estado" — o backup soma o valor da folha em todas as
// arestas do caminho, sem alternância de sinal.
//
// A avaliação de folha entra SÓ pela interface Evaluator (fronteira neutra): o
// núcleo abaixo é idêntico com rollout stub (Fase 1) ou rede (Fase 3/4).
// ============================================================================

import type { Action, Evaluator, GameState } from './types';
import { ACTIONS } from './types';
import { type RNG } from './rng';
import { applyMove, isTerminal, spawnOutcomes, withSpawn, type SpawnOutcome } from './board';
import { normalizeScore } from './evaluate';

// ----------------------------------------------------------------------------
// Nós da árvore
// ----------------------------------------------------------------------------

interface DecisionNode {
  readonly state: GameState;
  readonly terminal: boolean;
  expanded: boolean;
  /** Prior de política por ação (do Evaluator). */
  readonly P: Float64Array;
  /** Visitas por aresta de ação. */
  readonly N: Float64Array;
  /** Soma de valores retropropagados por aresta. */
  readonly W: Float64Array;
  /** Legalidade por ação (definida na expansão). */
  readonly legal: boolean[];
  /** Chance node por ação (lazy). */
  readonly children: (ChanceNode | null)[];
  /** Soma das visitas das arestas (= Σ N). */
  totalN: number;
}

interface ChanceNode {
  /** Board pós-movimento, antes do spawn. */
  readonly cells: Uint8Array;
  /** Pontos ganhos no movimento que criou este nó. */
  readonly gained: number;
  /** Score do estado de decisão pai (para compor o score dos filhos). */
  readonly baseScore: number;
  readonly size: number;
  /** Distribuição exata de spawn a partir de `cells`. */
  readonly outcomes: SpawnOutcome[];
  /** Nós de decisão filhos, por resultado de spawn (lazy). */
  readonly children: Map<number, DecisionNode>;
}

function createDecision(state: GameState): DecisionNode {
  return {
    state,
    terminal: isTerminal(state),
    expanded: false,
    P: new Float64Array(4),
    N: new Float64Array(4),
    W: new Float64Array(4),
    legal: [false, false, false, false],
    children: [null, null, null, null],
    totalN: 0,
  };
}

// ----------------------------------------------------------------------------
// Configuração e resultado
// ----------------------------------------------------------------------------

export interface MctsConfig {
  /** Simulações por busca. */
  readonly simulations: number;
  /** Coeficiente de exploração do PUCT. */
  readonly cPuct: number;
  /** Avaliador de folha (stub na Fase 1, rede na Fase 3/4). */
  readonly evaluator: Evaluator;
  /** RNG para amostragem dos chance nodes. */
  readonly rng: RNG;
  /**
   * Valor de um estado terminal. Padrão: normalizeScore (o valor verdadeiro de
   * um fim de jogo é seu score final normalizado — quantidade conhecida, não
   * aprendida).
   */
  readonly terminalValue?: (state: GameState) => number;
}

export interface SearchResult {
  /** Contagem de visitas por ação — a "política do MCTS". */
  readonly visits: number[];
  /** Q médio por ação (0 se não visitada). */
  readonly qValues: number[];
  /** Prior de política da raiz (do Evaluator). */
  readonly priors: number[];
  /** Ações legais na raiz. */
  readonly legal: boolean[];
  /** Ação mais visitada entre as legais (-1 se terminal). */
  readonly bestAction: Action | -1;
  /** Total de simulações efetivamente rodadas. */
  readonly simulations: number;
}

// ----------------------------------------------------------------------------
// Núcleo da busca
// ----------------------------------------------------------------------------

function expand(node: DecisionNode, evaluator: Evaluator): number {
  const evaluation = evaluator(node.state);
  const policy = evaluation.policy;
  for (const a of ACTIONS) {
    node.P[a] = policy[a] ?? 0;
    node.legal[a] = applyMove(node.state, a).moved;
  }
  node.expanded = true;
  return evaluation.value;
}

function selectAction(node: DecisionNode, cPuct: number): Action {
  const sqrtTotal = Math.sqrt(node.totalN);
  let best = -Infinity;
  let bestAction: Action = 0;
  for (const a of ACTIONS) {
    if (!node.legal[a]) continue;
    const n = node.N[a];
    const q = n > 0 ? node.W[a] / n : 0;
    const u = (cPuct * node.P[a] * sqrtTotal) / (1 + n);
    const s = q + u;
    if (s > best) {
      best = s;
      bestAction = a;
    }
  }
  return bestAction;
}

/** Chance node para a ação `a` (assume-se legal); criado sob demanda. */
function getChance(node: DecisionNode, a: Action): ChanceNode {
  const existing = node.children[a];
  if (existing) return existing;
  const res = applyMove(node.state, a);
  const chance: ChanceNode = {
    cells: res.cells,
    gained: res.gained,
    baseScore: node.state.score,
    size: node.state.size,
    outcomes: spawnOutcomes(res.cells),
    children: new Map(),
  };
  node.children[a] = chance;
  return chance;
}

/** Amostra um resultado de spawn ∝ probabilidade exata e desce para o filho. */
function sampleOutcome(chance: ChanceNode, rng: RNG): DecisionNode {
  const outcomes = chance.outcomes;
  let r = rng.next();
  let chosen = outcomes[outcomes.length - 1];
  for (const o of outcomes) {
    r -= o.prob;
    if (r < 0) {
      chosen = o;
      break;
    }
  }
  const key = chosen.index * 4 + chosen.exponent;
  const cached = chance.children.get(key);
  if (cached) return cached;
  const child = createDecision({
    size: chance.size,
    cells: withSpawn(chance.cells, chosen.index, chosen.exponent),
    score: chance.baseScore + chance.gained,
  });
  chance.children.set(key, child);
  return child;
}

interface PathStep {
  node: DecisionNode;
  action: Action;
}

function simulate(
  root: DecisionNode,
  evaluator: Evaluator,
  cPuct: number,
  rng: RNG,
  terminalValue: (state: GameState) => number,
): void {
  const path: PathStep[] = [];
  let node = root;
  let value: number;

  for (;;) {
    if (node.terminal) {
      value = terminalValue(node.state);
      break;
    }
    if (!node.expanded) {
      value = expand(node, evaluator);
      break;
    }
    const action = selectAction(node, cPuct);
    path.push({ node, action });
    const chance = getChance(node, action);
    node = sampleOutcome(chance, rng);
  }

  for (const step of path) {
    step.node.N[step.action] += 1;
    step.node.W[step.action] += value;
    step.node.totalN += 1;
  }
}

/** Roda a busca a partir de `rootState` e retorna as estatísticas de visita. */
export function runMcts(rootState: GameState, config: MctsConfig): SearchResult {
  const terminalValue = config.terminalValue ?? ((s: GameState) => normalizeScore(s.score));
  const root = createDecision(rootState);

  if (root.terminal) {
    return {
      visits: [0, 0, 0, 0],
      qValues: [0, 0, 0, 0],
      priors: [0, 0, 0, 0],
      legal: [false, false, false, false],
      bestAction: -1,
      simulations: 0,
    };
  }

  expand(root, config.evaluator);
  for (let s = 0; s < config.simulations; s++) {
    simulate(root, config.evaluator, config.cPuct, config.rng, terminalValue);
  }

  const visits = Array.from(root.N);
  const qValues = ACTIONS.map((a) => (root.N[a] > 0 ? root.W[a] / root.N[a] : 0));
  const priors = Array.from(root.P);

  let bestAction: Action | -1 = -1;
  let bestVisits = -1;
  for (const a of ACTIONS) {
    if (root.legal[a] && root.N[a] > bestVisits) {
      bestVisits = root.N[a];
      bestAction = a;
    }
  }

  return {
    visits,
    qValues,
    priors,
    legal: [...root.legal],
    bestAction,
    simulations: config.simulations,
  };
}

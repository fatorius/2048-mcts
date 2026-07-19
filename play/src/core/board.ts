// ============================================================================
// Motor do jogo 2048 — parametrizado por tamanho n desde já.
//
// Puro e determinístico onde possível: applyMove não spawna nem toca RNG. A
// aleatoriedade (spawn) fica isolada em funções que recebem um RNG, para manter
// a mecânica testável e os chance nodes do MCTS exatos.
// ============================================================================

import type { Action, GameState } from './types';
import { ACTIONS } from './types';
import { type RNG, randInt } from './rng';

/** Distribuição de spawn exata do 2048: expoente 1 ("2") vs 2 ("4"). */
export const SPAWN_PROB_2 = 0.9; // P(tile "2")
export const SPAWN_PROB_4 = 0.1; // P(tile "4")

// ----------------------------------------------------------------------------
// Índices de linha por ação (cacheados por tamanho)
// ----------------------------------------------------------------------------
//
// Para cada ação, o board é fatiado em `size` linhas de células, ordenadas da
// "cabeça" (para onde as peças deslizam) para a "cauda". Deslizar sempre em
// direção ao índice 0 de cada linha unifica as 4 direções.

const lineIndexCache = new Map<string, number[][]>();

function lineIndices(size: number, action: Action): number[][] {
  const key = `${size}:${action}`;
  const cached = lineIndexCache.get(key);
  if (cached) return cached;

  const idx = (r: number, c: number) => r * size + c;
  const lines: number[][] = [];
  for (let k = 0; k < size; k++) {
    const line: number[] = [];
    for (let j = 0; j < size; j++) {
      let r: number, c: number;
      switch (action) {
        case 0: // Up: coluna k, linhas de cima para baixo (cabeça no topo)
          r = j;
          c = k;
          break;
        case 1: // Right: linha k, colunas da direita para a esquerda
          r = k;
          c = size - 1 - j;
          break;
        case 2: // Down: coluna k, linhas de baixo para cima
          r = size - 1 - j;
          c = k;
          break;
        case 3: // Left: linha k, colunas da esquerda para a direita
          r = k;
          c = j;
          break;
      }
      line.push(idx(r, c));
    }
    lines.push(line);
  }
  lineIndexCache.set(key, lines);
  return lines;
}

// ----------------------------------------------------------------------------
// Mecânica de deslize + fusão
// ----------------------------------------------------------------------------

export interface MoveResult {
  /** Board após deslizar/fundir, ANTES do spawn. */
  readonly cells: Uint8Array;
  /** Pontos ganhos nas fusões deste movimento (soma dos valores fundidos). */
  readonly gained: number;
  /** Se o board mudou. Movimento ilegal <=> !moved. */
  readonly moved: boolean;
}

/**
 * Desliza uma linha de expoentes em direção ao índice 0, fundindo pares iguais
 * adjacentes (cada peça funde no máximo uma vez por movimento). Escreve o
 * resultado em `out` e retorna os pontos ganhos.
 */
function slideLine(vals: number[], out: number[]): number {
  let gained = 0;
  let write = 0;
  let prev = 0; // expoente pendente aguardando possível fusão (0 = nenhum)
  const n = vals.length;

  for (let i = 0; i < n; i++) {
    const v = vals[i];
    if (v === 0) continue;
    if (prev === v) {
      const merged = v + 1;
      out[write++] = merged;
      gained += 1 << merged; // valor da peça fundida = 2^merged
      prev = 0;
    } else {
      if (prev !== 0) out[write++] = prev;
      prev = v;
    }
  }
  if (prev !== 0) out[write++] = prev;
  while (write < n) out[write++] = 0;
  return gained;
}

/**
 * Aplica um movimento (deslizar + fundir) SEM spawnar. Puro: não muta o estado.
 * Base tanto do passo real quanto da expansão de chance nodes do MCTS.
 */
export function applyMove(state: GameState, action: Action): MoveResult {
  const { size, cells: src } = state;
  const cells = new Uint8Array(src);
  const buf: number[] = new Array(size);
  const out: number[] = new Array(size);
  let gained = 0;
  let moved = false;

  for (const line of lineIndices(size, action)) {
    for (let j = 0; j < size; j++) buf[j] = cells[line[j]];
    gained += slideLine(buf, out);
    for (let j = 0; j < size; j++) {
      const cell = line[j];
      if (cells[cell] !== out[j]) {
        cells[cell] = out[j];
        moved = true;
      }
    }
  }
  return { cells, gained, moved };
}

/** Movimento de uma peça que sobrevive ao deslize (para animação de UI). */
export interface TileMove {
  /** Célula de origem (a mais distante, no caso de fusão) — de onde desliza. */
  readonly from: number;
  /** Célula de destino. */
  readonly to: number;
  /** Expoente resultante (já +1 em fusões). */
  readonly exp: number;
  /** Se este destino é resultado de uma fusão. */
  readonly merged: boolean;
}

/**
 * Rastreia, para cada peça pós-deslize, de onde ela veio. Mesma lógica de
 * `applyMove` (sem spawn), mas retornando o mapeamento origem→destino que a UI
 * usa para animar. Não é usado pela busca.
 */
export function slideTiles(cells: Uint8Array, size: number, action: Action): TileMove[] {
  const moves: TileMove[] = [];
  for (const line of lineIndices(size, action)) {
    const entries: { cell: number; exp: number }[] = [];
    for (const idx of line) if (cells[idx] !== 0) entries.push({ cell: idx, exp: cells[idx] });
    let write = 0;
    let i = 0;
    while (i < entries.length) {
      const dest = line[write];
      if (i + 1 < entries.length && entries[i + 1].exp === entries[i].exp) {
        // Fusão: anima a partir da origem mais distante (a segunda na ordem).
        moves.push({ from: entries[i + 1].cell, to: dest, exp: entries[i].exp + 1, merged: true });
        i += 2;
      } else {
        moves.push({ from: entries[i].cell, to: dest, exp: entries[i].exp, merged: false });
        i += 1;
      }
      write++;
    }
  }
  return moves;
}

// ----------------------------------------------------------------------------
// Consultas de estado
// ----------------------------------------------------------------------------

export function emptyCells(state: GameState): number[] {
  const out: number[] = [];
  const { cells } = state;
  for (let i = 0; i < cells.length; i++) if (cells[i] === 0) out.push(i);
  return out;
}

export function legalActions(state: GameState): Action[] {
  const out: Action[] = [];
  for (const a of ACTIONS) if (applyMove(state, a).moved) out.push(a);
  return out;
}

/** Fim de jogo: nenhum movimento muda o board. */
export function isTerminal(state: GameState): boolean {
  const { size, cells } = state;
  // Qualquer célula vazia => existe deslize possível => não é terminal.
  for (let i = 0; i < cells.length; i++) if (cells[i] === 0) return false;
  // Board cheio: terminal a menos que haja par igual adjacente (horizontal/vertical).
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const v = cells[r * size + c];
      if (c + 1 < size && cells[r * size + c + 1] === v) return false;
      if (r + 1 < size && cells[(r + 1) * size + c] === v) return false;
    }
  }
  return true;
}

/** Maior expoente presente no board (0 se vazio). */
export function maxExponent(state: GameState): number {
  let m = 0;
  const { cells } = state;
  for (let i = 0; i < cells.length; i++) if (cells[i] > m) m = cells[i];
  return m;
}

// ----------------------------------------------------------------------------
// Spawn: distribuição exata e amostragem
// ----------------------------------------------------------------------------

export interface SpawnOutcome {
  /** Índice da célula (row-major) onde a peça aparece. */
  readonly index: number;
  /** Expoente da peça: 1 ("2") ou 2 ("4"). */
  readonly exponent: number;
  /** Probabilidade exata deste resultado dado o estado atual. */
  readonly prob: number;
}

/**
 * Enumeração exata de todos os spawns possíveis a partir de um board (usada
 * pelos chance nodes do MCTS). Para cada célula vazia, dois resultados: "2" com
 * peso 0.9 e "4" com peso 0.1, distribuídos uniformemente sobre as vazias. As
 * probabilidades somam 1.
 */
export function spawnOutcomes(cells: Uint8Array): SpawnOutcome[] {
  const empties: number[] = [];
  for (let i = 0; i < cells.length; i++) if (cells[i] === 0) empties.push(i);
  const e = empties.length;
  const out: SpawnOutcome[] = [];
  for (const index of empties) {
    out.push({ index, exponent: 1, prob: SPAWN_PROB_2 / e });
    out.push({ index, exponent: 2, prob: SPAWN_PROB_4 / e });
  }
  return out;
}

/** Clona `cells` e coloca `exponent` em `index`. */
export function withSpawn(cells: Uint8Array, index: number, exponent: number): Uint8Array {
  const next = new Uint8Array(cells);
  next[index] = exponent;
  return next;
}

/** Amostra um spawn segundo a distribuição exata, usando o RNG fornecido. */
export function spawnRandom(cells: Uint8Array, rng: RNG): Uint8Array {
  const empties: number[] = [];
  for (let i = 0; i < cells.length; i++) if (cells[i] === 0) empties.push(i);
  if (empties.length === 0) return new Uint8Array(cells);
  const index = empties[randInt(rng, empties.length)];
  const exponent = rng.next() < SPAWN_PROB_2 ? 1 : 2;
  return withSpawn(cells, index, exponent);
}

// ----------------------------------------------------------------------------
// Construção e passo do jogo
// ----------------------------------------------------------------------------

export function emptyState(size: number): GameState {
  return { size, cells: new Uint8Array(size * size), score: 0 };
}

/** Estado inicial: board vazio com 2 spawns, padrão 2048. */
export function initialState(size: number, rng: RNG): GameState {
  let cells: Uint8Array = new Uint8Array(size * size);
  cells = spawnRandom(cells, rng);
  cells = spawnRandom(cells, rng);
  return { size, cells, score: 0 };
}

/**
 * Passo real do jogo: aplica a ação e, se legal, spawna. Retorna o novo estado
 * e se o movimento foi legal. Ações ilegais retornam o mesmo estado com
 * moved=false.
 */
export function step(
  state: GameState,
  action: Action,
  rng: RNG,
): { state: GameState; moved: boolean } {
  const { cells, gained, moved } = applyMove(state, action);
  if (!moved) return { state, moved: false };
  return {
    state: { size: state.size, cells: spawnRandom(cells, rng), score: state.score + gained },
    moved: true,
  };
}

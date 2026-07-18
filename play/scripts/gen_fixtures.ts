// Gera fixtures de paridade do MOTOR a partir da implementação TS (fonte da
// verdade). O engine Python é verificado contra este JSON (train/tests/).
//
// Uso: npm run gen:fixtures  (escreve em train/tests/fixtures/engine_parity.json)

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  ACTIONS,
  applyMove,
  isTerminal,
  legalActions,
  normalizeScore,
  spawnOutcomes,
  type GameState,
} from '../src/core';
import { mulberry32, randInt } from '../src/core/rng';

const here = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(here, '../../train/tests/fixtures/engine_parity.json');

const rng = mulberry32(20482048);

/** Board aleatório de expoentes pequenos, com densidade variável. */
function randomBoard(size: number, maxExp: number, fillProb: number): GameState {
  const cells = new Uint8Array(size * size);
  for (let i = 0; i < cells.length; i++) {
    cells[i] = rng.next() < fillProb ? 1 + randInt(rng, maxExp) : 0;
  }
  const score = randInt(rng, 30000);
  return { size, cells, score };
}

interface Case {
  size: number;
  cells: number[];
  score: number;
  terminal: boolean;
  legal: number[];
  moves: Record<number, { cells: number[]; gained: number; moved: boolean }>;
  spawnOutcomes: { index: number; exponent: number; prob: number }[];
  normalizedScore: number;
}

const cases: Case[] = [];

for (const size of [3, 4, 5]) {
  for (let n = 0; n < 20; n++) {
    const fillProb = 0.3 + 0.65 * rng.next();
    const state = randomBoard(size, 6, fillProb);
    const moves: Case['moves'] = {};
    for (const a of ACTIONS) {
      const r = applyMove(state, a);
      moves[a] = { cells: Array.from(r.cells), gained: r.gained, moved: r.moved };
    }
    cases.push({
      size,
      cells: Array.from(state.cells),
      score: state.score,
      terminal: isTerminal(state),
      legal: legalActions(state),
      moves,
      spawnOutcomes: spawnOutcomes(state.cells).map((o) => ({
        index: o.index,
        exponent: o.exponent,
        prob: o.prob,
      })),
      normalizedScore: normalizeScore(state.score),
    });
  }
}

// Alguns boards deliberadamente cheios (testam terminal e fusões limítrofes).
const fullDense: number[][] = [
  [1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1], // xadrez 4x4 -> terminal
  [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 2], // com pares -> não terminal
];
for (const flat of fullDense) {
  const state: GameState = { size: 4, cells: Uint8Array.from(flat), score: 12345 };
  const moves: Case['moves'] = {};
  for (const a of ACTIONS) {
    const r = applyMove(state, a);
    moves[a] = { cells: Array.from(r.cells), gained: r.gained, moved: r.moved };
  }
  cases.push({
    size: 4,
    cells: flat,
    score: 12345,
    terminal: isTerminal(state),
    legal: legalActions(state),
    moves,
    spawnOutcomes: spawnOutcomes(state.cells).map((o) => ({
      index: o.index,
      exponent: o.exponent,
      prob: o.prob,
    })),
    normalizedScore: normalizeScore(state.score),
  });
}

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify({ cases }, null, 2));
console.log(`wrote ${cases.length} cases -> ${OUT}`);

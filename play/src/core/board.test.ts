import { describe, it, expect } from 'vitest';
import type { Action, GameState } from './types';
import {
  applyMove,
  isTerminal,
  legalActions,
  spawnOutcomes,
  spawnRandom,
  initialState,
  step,
  emptyCells,
} from './board';
import { mulberry32 } from './rng';

/** Constrói um estado a partir de uma grade de EXPOENTES (0 = vazio). */
function state(grid: number[][], score = 0): GameState {
  const size = grid.length;
  const cells = new Uint8Array(size * size);
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) cells[r * size + c] = grid[r][c];
  return { size, cells, score };
}

function grid(s: { size: number; cells: Uint8Array }): number[][] {
  const out: number[][] = [];
  for (let r = 0; r < s.size; r++) {
    const row: number[] = [];
    for (let c = 0; c < s.size; c++) row.push(s.cells[r * s.size + c]);
    out.push(row);
  }
  return out;
}

const UP = 0 as Action;
const RIGHT = 1 as Action;
const DOWN = 2 as Action;
const LEFT = 3 as Action;

describe('applyMove — deslize e fusão', () => {
  it('desliza para a esquerda sem fundir', () => {
    const s = state([
      [0, 1, 0, 2],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const r = applyMove(s, LEFT);
    expect(grid({ size: 4, cells: r.cells })[0]).toEqual([1, 2, 0, 0]);
    expect(r.moved).toBe(true);
    expect(r.gained).toBe(0);
  });

  it('funde um par igual e pontua 2^expoente', () => {
    const s = state([
      [1, 1, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const r = applyMove(s, LEFT);
    expect(grid({ size: 4, cells: r.cells })[0]).toEqual([2, 0, 0, 0]);
    expect(r.gained).toBe(4); // 2^2
  });

  it('funde no máximo um par por peça (2 2 2 2 -> 4 4)', () => {
    const s = state([
      [1, 1, 1, 1],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const r = applyMove(s, LEFT);
    expect(grid({ size: 4, cells: r.cells })[0]).toEqual([2, 2, 0, 0]);
    expect(r.gained).toBe(8); // duas fusões de 4
  });

  it('não funde em cadeia (4 2 2 -> 4 4, não 8)', () => {
    const s = state([
      [2, 1, 1, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const r = applyMove(s, LEFT);
    expect(grid({ size: 4, cells: r.cells })[0]).toEqual([2, 2, 0, 0]);
    expect(r.gained).toBe(4);
  });

  it('desliza para a direita', () => {
    const s = state([
      [1, 1, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const r = applyMove(s, RIGHT);
    expect(grid({ size: 4, cells: r.cells })[0]).toEqual([0, 0, 0, 2]);
  });

  it('desliza para cima e para baixo (colunas)', () => {
    const s = state([
      [1, 0, 0, 0],
      [1, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const up = applyMove(s, UP);
    expect(grid({ size: 4, cells: up.cells }).map((row) => row[0])).toEqual([2, 0, 0, 0]);
    const down = applyMove(s, DOWN);
    expect(grid({ size: 4, cells: down.cells }).map((row) => row[0])).toEqual([0, 0, 0, 2]);
  });

  it('movimento sem efeito é ilegal (moved=false)', () => {
    const s = state([
      [2, 1, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    expect(applyMove(s, LEFT).moved).toBe(false);
    expect(applyMove(s, RIGHT).moved).toBe(true);
  });

  it('não muta o estado de entrada', () => {
    const s = state([
      [1, 1, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const before = Array.from(s.cells);
    applyMove(s, LEFT);
    expect(Array.from(s.cells)).toEqual(before);
  });
});

describe('legalActions / isTerminal', () => {
  it('board com vazios nunca é terminal', () => {
    const s = state([
      [1, 2, 3, 4],
      [4, 3, 2, 1],
      [1, 2, 3, 4],
      [4, 3, 2, 0],
    ]);
    expect(isTerminal(s)).toBe(false);
  });

  it('board cheio sem pares adjacentes é terminal', () => {
    const s = state([
      [1, 2, 1, 2],
      [2, 1, 2, 1],
      [1, 2, 1, 2],
      [2, 1, 2, 1],
    ]);
    expect(isTerminal(s)).toBe(true);
    expect(legalActions(s)).toEqual([]);
  });

  it('board cheio com par adjacente não é terminal', () => {
    const s = state([
      [1, 1, 1, 2],
      [2, 1, 2, 1],
      [1, 2, 1, 2],
      [2, 1, 2, 1],
    ]);
    expect(isTerminal(s)).toBe(false);
    expect(legalActions(s).length).toBeGreaterThan(0);
  });
});

describe('distribuição de spawn', () => {
  it('spawnOutcomes soma 1 e cobre cada vazia com "2"/"4"', () => {
    const s = state([
      [1, 1, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const outcomes = spawnOutcomes(s.cells);
    const empties = emptyCells(s).length;
    expect(outcomes.length).toBe(empties * 2);
    const total = outcomes.reduce((a, o) => a + o.prob, 0);
    expect(total).toBeCloseTo(1, 12);
    // Razão 9:1 entre "2" e "4".
    const p2 = outcomes.filter((o) => o.exponent === 1).reduce((a, o) => a + o.prob, 0);
    expect(p2).toBeCloseTo(0.9, 12);
  });

  it('spawnRandom respeita ~90/10 sobre muitas amostras', () => {
    const rng = mulberry32(42);
    let twos = 0;
    const N = 20000;
    for (let i = 0; i < N; i++) {
      const cells = new Uint8Array(4); // tudo vazio
      const after = spawnRandom(cells, rng);
      const placed = Array.from(after).find((v) => v !== 0)!;
      if (placed === 1) twos++;
    }
    expect(twos / N).toBeGreaterThan(0.88);
    expect(twos / N).toBeLessThan(0.92);
  });
});

describe('inicialização e passo', () => {
  it('estado inicial tem exatamente 2 peças', () => {
    const s = initialState(4, mulberry32(1));
    const filled = Array.from(s.cells).filter((v) => v !== 0).length;
    expect(filled).toBe(2);
    expect(s.score).toBe(0);
  });

  it('step acumula score e adiciona uma peça', () => {
    const s = state([
      [1, 1, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const { state: next, moved } = step(s, LEFT, mulberry32(7));
    expect(moved).toBe(true);
    expect(next.score).toBe(4);
    // Antes: 2 peças; funde para 1; spawn adiciona 1 => 2 peças.
    expect(Array.from(next.cells).filter((v) => v !== 0).length).toBe(2);
  });

  it('reprodutível por seed', () => {
    const a = initialState(4, mulberry32(123));
    const b = initialState(4, mulberry32(123));
    expect(Array.from(a.cells)).toEqual(Array.from(b.cells));
  });
});

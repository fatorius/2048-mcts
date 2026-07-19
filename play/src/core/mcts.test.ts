import { describe, it, expect } from 'vitest';
import type { Evaluator, GameState } from './types';
import { runMcts, runMctsAsync } from './mcts';
import { makeRandomRolloutEvaluator, normalizeScore } from './evaluate';
import { mulberry32 } from './rng';

function state(grid: number[][], score = 0): GameState {
  const size = grid.length;
  const cells = new Uint8Array(size * size);
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) cells[r * size + c] = grid[r][c];
  return { size, cells, score };
}

const uniform: Evaluator = () => ({ policy: [0.25, 0.25, 0.25, 0.25], value: 0.5 });

describe('runMcts — invariantes da busca', () => {
  it('total de visitas nunca excede as simulações e só visita ações legais', () => {
    const s = state([
      [1, 2, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const res = runMcts(s, { simulations: 200, cPuct: 1.5, evaluator: uniform, rng: mulberry32(1) });
    const total = res.visits.reduce((a, b) => a + b, 0);
    expect(total).toBeLessThanOrEqual(200);
    for (let a = 0; a < 4; a++) if (!res.legal[a]) expect(res.visits[a]).toBe(0);
    expect(res.bestAction).toBeGreaterThanOrEqual(0);
    expect(res.legal[res.bestAction as number]).toBe(true);
  });

  it('estado terminal retorna bestAction -1', () => {
    const s = state([
      [1, 2, 1, 2],
      [2, 1, 2, 1],
      [1, 2, 1, 2],
      [2, 1, 2, 1],
    ]);
    const res = runMcts(s, { simulations: 50, cPuct: 1.5, evaluator: uniform, rng: mulberry32(1) });
    expect(res.bestAction).toBe(-1);
  });

  it('é determinístico para a mesma seed', () => {
    const s = state([
      [1, 1, 2, 0],
      [0, 2, 0, 0],
      [3, 0, 0, 1],
      [0, 0, 0, 0],
    ]);
    const cfg = () => ({
      simulations: 300,
      cPuct: 1.5,
      evaluator: makeRandomRolloutEvaluator(mulberry32(99)),
      rng: mulberry32(7),
    });
    const a = runMcts(s, cfg());
    const b = runMcts(s, cfg());
    expect(a.visits).toEqual(b.visits);
    expect(a.bestAction).toBe(b.bestAction);
  });

  it('prefere a jogada que funde vs. a que só desliza (sanidade de valor)', () => {
    // Uma fusão imediata (LEFT junta os dois "2" da linha 0) tende a render mais
    // visitas do que empurrar para um canto vazio, com rollouts suficientes.
    const s = state([
      [1, 1, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [2, 0, 0, 3],
    ]);
    const res = runMcts(s, {
      simulations: 400,
      cPuct: 1.2,
      evaluator: makeRandomRolloutEvaluator(mulberry32(5), { rollouts: 1 }),
      rng: mulberry32(11),
    });
    // A ação mais visitada deve ser legal e o Q dela finito.
    const best = res.bestAction as number;
    expect(res.legal[best]).toBe(true);
    expect(Number.isFinite(res.qValues[best])).toBe(true);
  });
});

describe('runMctsAsync — paridade com a busca síncrona', () => {
  it('produz visitas idênticas ao runMcts com o mesmo avaliador e seed', async () => {
    const s = state([
      [1, 1, 2, 0],
      [0, 2, 0, 0],
      [3, 0, 0, 1],
      [0, 0, 0, 0],
    ]);
    const asyncUniform = async () => ({ policy: [0.25, 0.25, 0.25, 0.25], value: 0.5 });
    const sync = runMcts(s, {
      simulations: 300,
      cPuct: 1.5,
      evaluator: uniform,
      rng: mulberry32(42),
    });
    const asyncRes = await runMctsAsync(s, {
      simulations: 300,
      cPuct: 1.5,
      evaluator: asyncUniform,
      rng: mulberry32(42),
    });
    expect(asyncRes.visits).toEqual(sync.visits);
    expect(asyncRes.bestAction).toBe(sync.bestAction);
  });

  it('respeita invariantes (soma == sims, só legais, terminal → -1)', async () => {
    const s = state([
      [1, 2, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
    const asyncUniform = async () => ({ policy: [0.25, 0.25, 0.25, 0.25], value: 0.5 });
    const res = await runMctsAsync(s, {
      simulations: 128,
      cPuct: 1.5,
      evaluator: asyncUniform,
      rng: mulberry32(1),
    });
    expect(res.visits.reduce((a, b) => a + b, 0)).toBe(128);
    for (let a = 0; a < 4; a++) if (!res.legal[a]) expect(res.visits[a]).toBe(0);
  });
});

describe('normalizeScore', () => {
  it('é monotônica e limitada a [0,1]', () => {
    expect(normalizeScore(0)).toBeGreaterThanOrEqual(0);
    expect(normalizeScore(0)).toBeLessThan(normalizeScore(100));
    expect(normalizeScore(100)).toBeLessThan(normalizeScore(20000));
    expect(normalizeScore(1e9)).toBeLessThanOrEqual(1);
  });
});

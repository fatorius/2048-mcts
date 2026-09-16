// Espelho de train/tests/test_reward_to_go.py — trava a paridade do backup
// reward-to-go do core TS com o do treino (Python).
import { describe, expect, it } from 'vitest';
import { makeValueTransform, runMcts } from './mcts';
import { mulberry32 } from './rng';
import { initialState } from './board';
import type { Evaluator } from './types';

describe('reward-to-go', () => {
  it('denorm∘renorm faz round-trip e renorm é monotônica em [0,1]', () => {
    const vt = makeValueTransform(1000, 500);
    for (const raw of [0, 250, 1000, 5000]) {
      expect(vt.denorm(vt.renorm(raw))).toBeCloseTo(raw, 2);
    }
    expect(vt.renorm(0)).toBeLessThan(vt.renorm(1000));
    expect(vt.renorm(1000)).toBeLessThan(vt.renorm(5000));
    // sem overflow nos extremos
    expect(vt.renorm(-1e9)).toBeGreaterThanOrEqual(0);
    expect(vt.renorm(1e9)).toBeLessThanOrEqual(1);
  });

  it('backup reward-to-go dá Q válido em [0,1] com avaliador constante', () => {
    const vt = makeValueTransform(1000, 500);
    const rng = mulberry32(1);
    const evaluator: Evaluator = () => ({ policy: [0.25, 0.25, 0.25, 0.25], value: 0.5 });
    const state = initialState(4, rng);
    const res = runMcts(state, {
      simulations: 200,
      cPuct: 1.5,
      evaluator,
      rng,
      valueTransform: vt,
    });
    expect(res.legal[res.bestAction as number]).toBe(true);
    for (const a of [0, 1, 2, 3]) {
      if (res.legal[a] && res.visits[a] > 0) {
        expect(res.qValues[a]).toBeGreaterThanOrEqual(0);
        expect(res.qValues[a]).toBeLessThanOrEqual(1);
      }
    }
  });
});

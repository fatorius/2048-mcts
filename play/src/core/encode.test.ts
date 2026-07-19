import { describe, it, expect } from 'vitest';
import fixtures from './__fixtures__/net_parity.json';
import { encode, NUM_CHANNELS } from './encode';
import type { GameState } from './types';

// Paridade do encoder TS contra o Python (encode.py). Fixtures geradas por
// `cd train && .venv/bin/python -m scripts.gen_net_fixtures <model.onnx>`.
// Tolerância 1e-6: os canais /17 são float32 no Python vs float64 no TS.

describe('encode — paridade exata com encode.py', () => {
  fixtures.cases.forEach((c, idx) => {
    it(`case ${idx} (n=${c.size})`, () => {
      const state: GameState = { size: c.size, cells: Uint8Array.from(c.cells), score: 0 };
      const x = encode(state);
      expect(x.length).toBe(c.encoding.length);
      expect(x.length).toBe(NUM_CHANNELS * c.size * c.size);
      let maxDiff = 0;
      for (let i = 0; i < x.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(x[i] - c.encoding[i]));
      }
      expect(maxDiff).toBeLessThan(1e-6);
    });
  });
});

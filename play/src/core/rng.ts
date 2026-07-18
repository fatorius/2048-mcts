// Gerador pseudoaleatório seedável e determinístico.
//
// Necessário para reprodutibilidade: os testes de paridade da Fase 3 dependem de
// que spawns e rollouts sejam replicáveis a partir de uma seed. Math.random não
// é seedável, então threadamos uma instância de RNG por spawn/rollout/seleção.

export interface RNG {
  /** Próximo float em [0, 1). */
  next(): number;
}

/** mulberry32 — PRNG rápido de 32 bits, boa distribuição para simulação. */
export function mulberry32(seed: number): RNG {
  let a = seed >>> 0;
  return {
    next(): number {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
  };
}

/** Inteiro uniforme em [0, n). */
export function randInt(rng: RNG, n: number): number {
  return Math.floor(rng.next() * n);
}

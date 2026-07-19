// Codificação do board no tensor de entrada 20×n×n — ESPELHO EXATO de
// train/twenty48/encode.py. Precisa bater bit-a-bit (dentro de float32) com o
// Python, senão a rede recebe entrada errada e joga mal (bug silencioso).
// Verificado por fixtures: encode.test.ts.
//
// Layout NCHW achatado em C-order: índice = canal*n*n + (linha*n + coluna).
//   canais 0..16 : one-hot por expoente e∈[1..17] (clipado), canal e-1
//   canal   17   : vazio (célula == 0)
//   canal   18   : expoente bruto / 17
//   canal   19   : expoente máximo do board / 17 (broadcast)

import type { GameState } from './types';

export const NUM_CHANNELS = 20;
export const MAX_EXPONENT = 17;

/** Retorna Float32Array de comprimento 20*n*n, pronto para o tensor ORT [1,20,n,n]. */
export function encode(state: GameState): Float32Array {
  const { size: n, cells } = state;
  const area = n * n;
  const x = new Float32Array(NUM_CHANNELS * area);

  let maxExp = 0;
  for (let i = 0; i < area; i++) if (cells[i] > maxExp) maxExp = cells[i];
  const maxNorm = maxExp / MAX_EXPONENT;

  for (let i = 0; i < area; i++) {
    const v = cells[i];
    const clipped = v > MAX_EXPONENT ? MAX_EXPONENT : v;
    if (clipped >= 1) x[(clipped - 1) * area + i] = 1; // one-hot
    if (v === 0) x[17 * area + i] = 1; // vazio
    x[18 * area + i] = v / MAX_EXPONENT; // magnitude bruta
    x[19 * area + i] = maxNorm; // magnitude máxima (broadcast)
  }
  return x;
}

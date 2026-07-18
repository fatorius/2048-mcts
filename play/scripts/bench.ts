// ============================================================================
// Critério de saída da Fase 1 (mensurável):
// MCTS puro com rollouts aleatórios deve alcançar 2048 numa fração razoável das
// partidas 4×4. Se não alcançar, há bug na busca — pegar agora, antes da rede.
//
// Uso: npm run bench -- [games] [simulations] [size] [cPuct] [seed]
// ============================================================================

import { mulberry32 } from '../src/core/rng';
import { makeRandomRolloutEvaluator } from '../src/core/evaluate';
import { playGame } from '../src/core/agent';

const args = process.argv.slice(2);
const games = Number(args[0] ?? 30);
const simulations = Number(args[1] ?? 300);
const size = Number(args[2] ?? 4);
const cPuct = Number(args[3] ?? 1.5);
const baseSeed = Number(args[4] ?? 1);

console.log(
  `2048 MCTS bench — size=${size}, games=${games}, sims/move=${simulations}, cPuct=${cPuct}\n`,
);

const tileCounts = new Map<number, number>();
let totalScore = 0;
let best = 0;
let reached2048 = 0;
const t0 = Date.now();

for (let g = 0; g < games; g++) {
  const rng = mulberry32(baseSeed + g * 1000);
  const result = playGame(
    size,
    { simulations, cPuct, makeEvaluator: (r) => makeRandomRolloutEvaluator(r) },
    rng,
  );
  const tile = result.maxTile;
  tileCounts.set(tile, (tileCounts.get(tile) ?? 0) + 1);
  totalScore += result.score;
  if (tile > best) best = tile;
  if (result.maxExponent >= 11) reached2048++;
  process.stdout.write(
    `  game ${String(g + 1).padStart(3)}  score=${String(result.score).padStart(6)}  ` +
      `maxTile=${String(tile).padStart(5)}  moves=${result.moves}\n`,
  );
}

const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
console.log('\n' + '─'.repeat(48));
console.log(`Distribuição de maior peça (${games} partidas):`);
for (const tile of [...tileCounts.keys()].sort((a, b) => a - b)) {
  const n = tileCounts.get(tile)!;
  const pct = ((100 * n) / games).toFixed(0);
  console.log(`  ${String(tile).padStart(5)}: ${'█'.repeat(n)} ${n} (${pct}%)`);
}
console.log('─'.repeat(48));
console.log(`Score médio : ${(totalScore / games).toFixed(0)}`);
console.log(`Melhor peça : ${best}`);
console.log(
  `Alcançou 2048: ${reached2048}/${games} (${((100 * reached2048) / games).toFixed(0)}%)`,
);
console.log(`Tempo       : ${elapsed}s`);

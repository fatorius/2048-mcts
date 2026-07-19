import type { CSSProperties } from 'react';
import type { Action, GameState } from '../core';
import { slideTiles } from '../core';

/** Cor da peça por expoente — paleta quente que escurece conforme cresce. */
function tileStyle(exp: number): CSSProperties {
  const hue = ((45 - exp * 14) % 360 + 360) % 360;
  const light = Math.max(42, 80 - exp * 3.2);
  return {
    background: `hsl(${hue} 68% ${light}%)`,
    color: light > 58 ? '#2b2018' : '#fff8f0',
  };
}

type Kind = 'static' | 'slide' | 'merge' | 'spawn';

interface Sprite {
  cell: number;
  exp: number;
  kind: Kind;
  dx: number; // deslocamento origem→destino em células (para o slide)
  dy: number;
}

/**
 * Deriva as peças a animar comparando o estado anterior (via ação tomada) com o
 * atual: peças que deslizaram, fusões, e a peça nova (spawn). Sem `prevCells`
 * (nova partida), tudo entra com pop.
 */
function computeSprites(state: GameState, prevCells: Uint8Array | null, action: Action | -1): Sprite[] {
  const { size, cells } = state;
  if (prevCells === null || action === -1) {
    const out: Sprite[] = [];
    for (let i = 0; i < cells.length; i++) {
      if (cells[i] !== 0) out.push({ cell: i, exp: cells[i], kind: 'spawn', dx: 0, dy: 0 });
    }
    return out;
  }

  const moves = slideTiles(prevCells, size, action);
  const covered = new Set(moves.map((m) => m.to));
  const sprites: Sprite[] = moves.map((m) => {
    const fr = Math.floor(m.from / size);
    const fc = m.from % size;
    const tr = Math.floor(m.to / size);
    const tc = m.to % size;
    const kind: Kind = m.merged ? 'merge' : m.from === m.to ? 'static' : 'slide';
    return { cell: m.to, exp: m.exp, kind, dx: fc - tc, dy: fr - tr };
  });
  // A peça nova é a célula ocupada do estado que o deslize não produziu.
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] !== 0 && !covered.has(i)) {
      sprites.push({ cell: i, exp: cells[i], kind: 'spawn', dx: 0, dy: 0 });
    }
  }
  return sprites;
}

export function Board({
  state,
  prevCells,
  action,
  moveId,
}: {
  state: GameState;
  prevCells: Uint8Array | null;
  action: Action | -1;
  moveId: number;
}) {
  const { size } = state;
  const sprites = computeSprites(state, prevCells, action);

  return (
    <div className="board" style={{ '--n': size } as CSSProperties}>
      {Array.from({ length: size * size }, (_, i) => (
        <div
          key={`bg${i}`}
          className="cell-bg"
          style={{ gridColumn: (i % size) + 1, gridRow: Math.floor(i / size) + 1 }}
        />
      ))}
      {sprites.map((s) => (
        <div
          key={`${moveId}:${s.cell}`}
          className={`tile anim-${s.kind}`}
          data-len={String(1 << s.exp).length}
          style={
            {
              gridColumn: (s.cell % size) + 1,
              gridRow: Math.floor(s.cell / size) + 1,
              '--dx': s.dx,
              '--dy': s.dy,
              ...tileStyle(s.exp),
            } as CSSProperties
          }
        >
          {1 << s.exp}
        </div>
      ))}
    </div>
  );
}

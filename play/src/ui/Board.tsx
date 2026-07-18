import type { CSSProperties } from 'react';
import type { GameState } from '../core';

/** Cor da peça por expoente — paleta quente que escurece conforme cresce. */
function tileStyle(exp: number): CSSProperties {
  if (exp === 0) return {};
  const hue = ((45 - exp * 14) % 360 + 360) % 360;
  const light = Math.max(42, 80 - exp * 3.2);
  return {
    background: `hsl(${hue} 68% ${light}%)`,
    color: light > 58 ? '#2b2018' : '#fff8f0',
  };
}

export function Board({ state }: { state: GameState }) {
  const { size, cells } = state;
  return (
    <div className="board" style={{ '--n': size } as CSSProperties}>
      {Array.from(cells).map((exp, i) => (
        <div
          key={i}
          className={exp === 0 ? 'tile empty' : 'tile'}
          style={tileStyle(exp)}
          data-len={exp === 0 ? 0 : String(1 << exp).length}
        >
          {exp === 0 ? '' : 1 << exp}
        </div>
      ))}
    </div>
  );
}

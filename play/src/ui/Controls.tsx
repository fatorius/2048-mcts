import type { Controller } from '../game/useGameController';

const SIZES = [3, 4, 5, 6];
const SIM_OPTIONS = [25, 50, 100, 200, 400, 800, 1600];

export function Controls({ c }: { c: Controller }) {
  const playing = c.status === 'playing';
  const thinking = c.status === 'thinking';
  const over = c.status === 'gameover';
  const hasMove = !!c.result && c.result.bestAction !== -1;

  return (
    <div className="controls">
      <div className="btn-row">
        <button className="btn primary" onClick={c.newGame}>
          New game
        </button>
        {playing ? (
          <button className="btn" onClick={c.pause}>
            ⏸ Pause
          </button>
        ) : (
          <button className="btn" onClick={c.play} disabled={!hasMove || over}>
            ▶ Play
          </button>
        )}
        <button className="btn" onClick={c.step} disabled={playing || thinking || !hasMove || over}>
          ⏭ Step
        </button>
      </div>

      <label className="field">
        <span>Board size</span>
        <select value={c.config.size} onChange={(e) => c.setSize(Number(e.target.value))}>
          {SIZES.map((n) => (
            <option key={n} value={n}>
              {n}×{n}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>
          Simulations / move <b>{c.config.simulations}</b>
        </span>
        <input
          type="range"
          min={0}
          max={SIM_OPTIONS.length - 1}
          step={1}
          value={Math.max(0, SIM_OPTIONS.indexOf(c.config.simulations))}
          onChange={(e) => c.setSimulations(SIM_OPTIONS[Number(e.target.value)])}
        />
      </label>

      <label className="field">
        <span>
          c_puct <b>{c.config.cPuct.toFixed(1)}</b>
        </span>
        <input
          type="range"
          min={0.5}
          max={4}
          step={0.1}
          value={c.config.cPuct}
          onChange={(e) => c.setCPuct(Number(e.target.value))}
        />
      </label>

      <label className="field">
        <span>
          Move delay <b>{c.config.delayMs} ms</b>
        </span>
        <input
          type="range"
          min={0}
          max={600}
          step={20}
          value={c.config.delayMs}
          onChange={(e) => c.setDelayMs(Number(e.target.value))}
        />
      </label>
    </div>
  );
}

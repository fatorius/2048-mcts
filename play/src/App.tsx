import { useGameController, type Status } from './game/useGameController';
import { Board } from './ui/Board';
import { VisitStats } from './ui/VisitStats';
import { Controls } from './ui/Controls';
import { maxExponent } from './core';
import './App.css';

const STATUS_LABEL: Record<Status, string> = {
  loading: 'Loading…',
  idle: 'Ready',
  thinking: 'Thinking…',
  playing: 'Playing',
  gameover: 'Game over',
};

function App() {
  const c = useGameController();
  const maxTile = c.state ? 1 << maxExponent(c.state) : 0;

  return (
    <div className="app">
      <header className="app-header">
        <h1>2048 · MCTS</h1>
        <p className="subtitle">
          Busca pura (rollouts aleatórios) — a janela de depuração da Fase 1
        </p>
      </header>

      <main className="layout">
        <section className="board-col">
          {c.state ? (
            <Board
              state={c.state}
              prevCells={c.prevCells}
              action={c.lastAction}
              moveId={c.moves}
            />
          ) : (
            <div className="board-loading">Loading…</div>
          )}

          <div className="game-stats">
            <div className="stat">
              <span className="stat-k">Score</span>
              <span className="stat-v">{c.state?.score ?? 0}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Max tile</span>
              <span className="stat-v">{maxTile}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Moves</span>
              <span className="stat-v">{c.moves}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Status</span>
              <span className={`stat-v status ${c.status}`}>{STATUS_LABEL[c.status]}</span>
            </div>
          </div>
        </section>

        <aside className="side">
          <Controls c={c} />

          <div className="panel">
            <div className="panel-head">
              <h2>MCTS visit stats</h2>
              <span className="panel-meta">
                {c.lastElapsedMs != null ? `${c.lastElapsedMs.toFixed(1)} ms` : '—'} ·{' '}
                {c.config.simulations} sims
              </span>
            </div>
            <VisitStats result={c.result} />
            <p className="panel-note">
              A barra é a distribuição de visitas (a “política do MCTS”). Q = valor médio, P =
              prior. A direção destacada é a jogada escolhida; direções apagadas são ilegais.
            </p>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;

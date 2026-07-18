import type { SearchResult } from '../core';
import { ACTION_NAMES } from '../core';

// A janela para dentro da busca: quantas simulações cada direção recebeu, com Q
// (valor médio) e P (prior). É o diagnóstico central da Fase 2 — e na Fase 3
// passa a exibir a política prevista da rede lado a lado com estas visitas.

const ARROWS = ['↑', '→', '↓', '←'] as const;

export function VisitStats({ result }: { result: SearchResult | null }) {
  const total = result ? result.visits.reduce((a, b) => a + b, 0) || 1 : 1;
  return (
    <div className="visit-stats">
      {[0, 1, 2, 3].map((a) => {
        const visits = result ? result.visits[a] : 0;
        const pct = result ? (100 * visits) / total : 0;
        const legal = result ? result.legal[a] : false;
        const best = result ? a === result.bestAction : false;
        const cls = ['vs-row', legal ? '' : 'illegal', best ? 'best' : ''].filter(Boolean).join(' ');
        return (
          <div key={a} className={cls}>
            <span className="vs-dir">
              <span className="vs-arrow">{ARROWS[a]}</span>
              {ACTION_NAMES[a]}
            </span>
            <div className="vs-bar-wrap">
              <div className="vs-bar" style={{ width: `${pct}%` }} />
            </div>
            <span className="vs-visits">{result ? visits : '—'}</span>
            <span className="vs-num">Q {result ? result.qValues[a].toFixed(3) : '—'}</span>
            <span className="vs-num">P {result ? result.priors[a].toFixed(2) : '—'}</span>
          </div>
        );
      })}
    </div>
  );
}

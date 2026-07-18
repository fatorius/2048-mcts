// ============================================================================
// CONTRATOS CONGELADOS (Fase 1 usa como stub; Fase 3/4 implementam de verdade)
// ----------------------------------------------------------------------------
// Estes dois contratos são a fronteira compartilhada entre a busca (JS/TS), a
// codificação de entrada da rede (Python) e a serialização entre runtimes.
// NÃO alterar sem migrar todas as fases. Ver plano_2048_mcts_rl.md §"Contratos".
// ============================================================================

/**
 * Ações do jogador. Ordem fixa e canônica — a cabeça de política da rede
 * (Fase 3) produz 4 logits nesta mesma ordem.
 *   0 = Up, 1 = Right, 2 = Down, 3 = Left
 */
export type Action = 0 | 1 | 2 | 3;

export const ACTIONS: readonly Action[] = [0, 1, 2, 3];

export const ACTION_NAMES = ['Up', 'Right', 'Down', 'Left'] as const;

// ----------------------------------------------------------------------------
// Contrato 1 — Formato de estado do tabuleiro
// ----------------------------------------------------------------------------
//
// Representação canônica de um board. `cells` guarda o EXPOENTE de cada célula
// (log2 do valor), não o valor:
//     0  -> célula vazia
//     1  -> tile "2"
//     2  -> tile "4"
//     k  -> tile 2^k
//
// Layout row-major: a célula (linha r, coluna c) fica em `cells[r * size + c]`.
//
// Por que expoentes e não valores: a entrada da rede (Fase 3) é um one-hot por
// expoente. Guardar o expoente aqui torna a codificação um mapeamento direto e
// a serialização Python↔JS trivial (bytes 0..17).
// ----------------------------------------------------------------------------

export interface GameState {
  /** n, a dimensão do tabuleiro n×n. O núcleo é agnóstico a n. */
  readonly size: number;
  /** Expoentes row-major, comprimento size*size. 0 = vazio. */
  readonly cells: Uint8Array;
  /** Pontuação acumulada (soma dos valores das fusões), padrão 2048. */
  readonly score: number;
}

// ----------------------------------------------------------------------------
// Contrato 2 — Interface evaluate(state) -> (policy_prior, value)
// ----------------------------------------------------------------------------
//
// Função plugável que o MCTS chama nas folhas. É a fronteira neutra que isola a
// busca de COMO a avaliação é produzida:
//   - Fase 1 (aqui): prior uniforme + valor por rollout aleatório/heurística.
//   - Fase 3/4:      a rede (forward pass) substitui o stub, sem tocar o MCTS.
// ----------------------------------------------------------------------------

export interface Evaluation {
  /**
   * Prior de política sobre ACTIONS (comprimento 4, soma ≈ 1). Guia a expansão
   * do MCTS. No stub da Fase 1 é uniforme; na Fase 3 é a saída da cabeça de
   * política.
   */
  readonly policy: readonly number[];
  /**
   * Estimativa escalar da qualidade do estado, em [0, 1] (maior = melhor).
   * O MCTS faz a média deste valor ao longo das simulações. A cabeça de valor
   * da rede (Fase 3) prevê o mesmo alvo normalizado.
   */
  readonly value: number;
}

export type Evaluator = (state: GameState) => Evaluation;

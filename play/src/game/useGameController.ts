// Controlador da partida (main thread). Mantém o estado autoritativo em refs
// para o laço imperativo de auto-play sobre o worker assíncrono, e espelha em
// estado React para renderizar. Um contador de "geração" cancela laços obsoletos
// quando uma nova partida começa no meio de um play.

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Action, GameState, SearchResult } from '../core';
import { initialState, isTerminal, mulberry32, step, type RNG } from '../core';
import { SearchClient } from '../search/searchClient';

export type Status = 'loading' | 'idle' | 'thinking' | 'playing' | 'gameover';
export type NetStatus = 'off' | 'loading' | 'ready' | 'error';

const MODEL_URL = '/model.onnx';

export interface GameConfig {
  size: number;
  simulations: number;
  cPuct: number;
  delayMs: number;
  /** Usar a rede ONNX (true) ou o stub de rollout (false) como avaliador. */
  useNet: boolean;
}

const DEFAULT_CONFIG: GameConfig = {
  size: 4,
  simulations: 200,
  cPuct: 1.5,
  delayMs: 120,
  useNet: false,
};

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export interface Controller {
  state: GameState | null;
  result: SearchResult | null;
  status: Status;
  moves: number;
  lastElapsedMs: number | null;
  config: GameConfig;
  netStatus: NetStatus;
  netBackend: string | null;
  /** Board antes do último lance (para animação); null em nova partida. */
  prevCells: Uint8Array | null;
  /** Ação do último lance (para animação); -1 se nenhum. */
  lastAction: Action | -1;
  newGame: () => void;
  play: () => void;
  pause: () => void;
  step: () => void;
  setSize: (n: number) => void;
  setSimulations: (n: number) => void;
  setCPuct: (c: number) => void;
  setDelayMs: (d: number) => void;
  setUseNet: (v: boolean) => void;
}

interface RenderState {
  state: GameState | null;
  result: SearchResult | null;
  status: Status;
  moves: number;
  lastElapsedMs: number | null;
  netStatus: NetStatus;
  netBackend: string | null;
  prevCells: Uint8Array | null;
  lastAction: Action | -1;
}

const INITIAL_RENDER: RenderState = {
  state: null,
  result: null,
  status: 'loading',
  moves: 0,
  lastElapsedMs: null,
  netStatus: 'off',
  netBackend: null,
  prevCells: null,
  lastAction: -1,
};

export function useGameController(): Controller {
  const [config, setConfig] = useState<GameConfig>(DEFAULT_CONFIG);
  // Espelho de render: os refs são a fonte da verdade do laço; isto é o que a UI
  // lê (nunca acessamos refs durante o render).
  const [render, setRender] = useState<RenderState>(INITIAL_RENDER);

  // Estado autoritativo (refs) — fonte da verdade para o laço.
  const clientRef = useRef<SearchClient | null>(null);
  const rngRef = useRef<RNG>(mulberry32(1));
  const stateRef = useRef<GameState | null>(null);
  const resultRef = useRef<SearchResult | null>(null);
  const playingRef = useRef(false);
  const busyRef = useRef(false);
  const genRef = useRef(0);
  const seedRef = useRef(1);
  const statusRef = useRef<Status>('loading');
  const elapsedRef = useRef<number | null>(null);
  const movesRef = useRef(0);
  const prevCellsRef = useRef<Uint8Array | null>(null);
  const lastActionRef = useRef<Action | -1>(-1);
  const netReadyRef = useRef(false);
  const netStatusRef = useRef<NetStatus>('off');
  const netBackendRef = useRef<string | null>(null);
  const configRef = useRef(config);
  useEffect(() => {
    configRef.current = config;
  }, [config]);

  // Lê os refs (fora do render) e publica no espelho de estado.
  const commit = useCallback(() => {
    setRender({
      state: stateRef.current,
      result: resultRef.current,
      status: statusRef.current,
      moves: movesRef.current,
      lastElapsedMs: elapsedRef.current,
      netStatus: netStatusRef.current,
      netBackend: netBackendRef.current,
      prevCells: prevCellsRef.current,
      lastAction: lastActionRef.current,
    });
  }, []);

  const setStatus = useCallback(
    (s: Status) => {
      statusRef.current = s;
      commit();
    },
    [commit],
  );

  /** Busca a partir de `s`; devolve o SearchResult ou null se ficou obsoleto. */
  const searchFrom = useCallback(async (s: GameState, gen: number): Promise<SearchResult | null> => {
    const cfg = configRef.current;
    // Modo rede só se o usuário ligou E o modelo já carregou; senão, rollout.
    const mode = cfg.useNet && netReadyRef.current ? 'net' : 'rollout';
    const outcome = await clientRef.current!.search(s, cfg.simulations, cfg.cPuct, mode);
    if (gen !== genRef.current) return null;
    resultRef.current = outcome.result;
    elapsedRef.current = outcome.elapsedMs;
    return outcome.result;
  }, []);

  /** Aplica um lance (best move + spawn real) e busca a partir do novo estado. */
  const driveOneMove = useCallback(
    async (gen: number): Promise<boolean> => {
      const r = resultRef.current;
      const cur = stateRef.current;
      if (!r || r.bestAction === -1 || !cur) return false;

      const { state: next } = step(cur, r.bestAction as Action, rngRef.current);
      if (gen !== genRef.current) return false;
      // Info para a animação: de onde as peças vieram + a direção.
      prevCellsRef.current = cur.cells;
      lastActionRef.current = r.bestAction as Action;
      stateRef.current = next;
      movesRef.current += 1;

      if (isTerminal(next)) {
        resultRef.current = null;
        playingRef.current = false;
        setStatus('gameover');
        return false;
      }
      commit();
      const searched = await searchFrom(next, gen);
      if (!searched) return false;
      commit();
      return true;
    },
    [commit, searchFrom, setStatus],
  );

  const newGame = useCallback(() => {
    const gen = ++genRef.current;
    playingRef.current = false;
    busyRef.current = false;
    const cfg = configRef.current;
    const seed = (seedRef.current += 1);
    clientRef.current!.reset(seed);
    rngRef.current = mulberry32((seed ^ 0x9e3779b9) >>> 0);
    const s0 = initialState(cfg.size, rngRef.current);
    stateRef.current = s0;
    resultRef.current = null;
    elapsedRef.current = null;
    movesRef.current = 0;
    prevCellsRef.current = null; // nova partida: peças iniciais entram com pop
    lastActionRef.current = -1;
    setStatus('thinking');
    void searchFrom(s0, gen).then((r) => {
      if (r) setStatus('idle');
    });
  }, [searchFrom, setStatus]);

  const step_ = useCallback(() => {
    if (playingRef.current || busyRef.current) return;
    if (!resultRef.current || resultRef.current.bestAction === -1) return;
    const gen = genRef.current;
    busyRef.current = true;
    setStatus('thinking');
    void driveOneMove(gen).then((cont) => {
      busyRef.current = false;
      if (gen !== genRef.current) return;
      if (cont) setStatus('idle');
    });
  }, [driveOneMove, setStatus]);

  const play = useCallback(() => {
    if (playingRef.current || busyRef.current) return;
    if (!resultRef.current || resultRef.current.bestAction === -1) return;
    const gen = genRef.current;
    playingRef.current = true;
    setStatus('playing');
    void (async () => {
      while (playingRef.current && gen === genRef.current) {
        const cont = await driveOneMove(gen);
        if (!cont) break;
        const delay = configRef.current.delayMs;
        if (delay > 0) await sleep(delay);
      }
      if (gen === genRef.current && statusRef.current !== 'gameover') {
        playingRef.current = false;
        setStatus('idle');
      }
    })();
  }, [driveOneMove, setStatus]);

  const pause = useCallback(() => {
    playingRef.current = false;
    if (statusRef.current === 'playing') setStatus('idle');
  }, [setStatus]);

  // Ciclo de vida do worker + primeira partida.
  useEffect(() => {
    const client = new SearchClient(seedRef.current);
    clientRef.current = client;
    newGame();
    return () => {
      genRef.current += 1;
      playingRef.current = false;
      client.terminate();
      clientRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Trocar o tamanho reinicia a partida (a dimensão do board muda). Pula o mount
  // (a primeira partida já é criada no efeito acima).
  const sizeInitRef = useRef(true);
  useEffect(() => {
    if (sizeInitRef.current) {
      sizeInitRef.current = false;
      return;
    }
    if (clientRef.current) newGame();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.size]);

  const setSize = useCallback((n: number) => setConfig((c) => ({ ...c, size: n })), []);
  const setSimulations = useCallback((n: number) => setConfig((c) => ({ ...c, simulations: n })), []);
  const setCPuct = useCallback((cp: number) => setConfig((c) => ({ ...c, cPuct: cp })), []);
  const setDelayMs = useCallback((d: number) => setConfig((c) => ({ ...c, delayMs: d })), []);

  const setUseNet = useCallback(
    (v: boolean) => {
      setConfig((c) => ({ ...c, useNet: v }));
      // Carrega o modelo sob demanda na primeira vez que liga (ou após erro).
      if (v && (netStatusRef.current === 'off' || netStatusRef.current === 'error')) {
        netStatusRef.current = 'loading';
        commit();
        clientRef.current?.loadNet(MODEL_URL).then((r) => {
          netReadyRef.current = r.ok;
          netStatusRef.current = r.ok ? 'ready' : 'error';
          netBackendRef.current = r.ok ? (r.backend ?? null) : (r.error ?? 'erro');
          commit();
        });
      }
    },
    [commit],
  );

  return {
    state: render.state,
    result: render.result,
    status: render.status,
    moves: render.moves,
    lastElapsedMs: render.lastElapsedMs,
    config,
    netStatus: render.netStatus,
    netBackend: render.netBackend,
    prevCells: render.prevCells,
    lastAction: render.lastAction,
    newGame,
    play,
    pause,
    step: step_,
    setSize,
    setSimulations,
    setCPuct,
    setDelayMs,
    setUseNet,
  };
}

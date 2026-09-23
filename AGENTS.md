# AGENTS.md

Guidance for AI agents working on this repo (2048 — MCTS guided by an offline-RL-trained
two-headed CNN; `train/` = Python training, chance-node MCTS, size-agnostic net).

## Performance profile & scaling plan (measured 2026-09-22)

Where self-play time actually goes, why the GPU looks under-used, and what changes if the
training hardware improves. All numbers measured on the current box (see
`train/` and the memory note "gpu-training-box"): **GTX 750 Ti, Maxwell sm_50, 38.5 W power
cap, ~1.9 GB free VRAM.** Net = **128 ch × 6 blocks, ~779k params.**

### Pure-GPU forward ceiling (net forward only, no CPU in the loop, 6×6)

| batch | ms/call | states/s | note |
|------:|--------:|---------:|------|
| 6     | 0.91    | 6,581    | latency-bound — SMs starved by tiny batch |
| 64    | 3.90    | 16,417   | ~96 % of ceiling |
| 256   | 14.94   | 17,136   | ceiling |
| 1024  | 115.4   | 8,875    | **regresses** — exceeds ~1.9 GB VRAM, thrashes |

**GPU ceiling ≈ 17k states/s; saturates at batch ~64; do NOT exceed ~256 on this card.**

### Real self-play cost split (batch 64, 6×6, sims 400)

Per-state: GPU forward **66 µs**, encode (CPU) **35 µs**, copy 1 µs,
**MCTS tree + `board.step` (CPU, single-thread) 64 µs**. CPU subtotal ≈ **99 µs/state →
~10,100 states/s per core.** Measured overall self-play throughput was only
**~3,300 st/s (batch 6) / ~6,000 st/s (batch 64)** — far below the 17k GPU ceiling.

### Diagnosis: the ~50 % GPU util is CPU-STARVATION, not throttling

Confirmed, not inferred. Fed back-to-back the card hits **100 % util, boosts to 1280 MHz
(idle 135, max 1450), draws the full 38.5 W, at 50 °C** (slowdown 96 °C). Every
`nvidia-smi` throttle reason is "Not Active" except `Idle`. The loss is idle gaps: the
synchronous driver does `forward → (GPU idle while single-threaded CPU does encode+tree) →
forward`. Util even *fell* 51 %→39 % at batch 64 because CPU work per round grew.

### Two throughput levers (same underlying fix)

1. **More concurrent games** → bigger batch (raises the forward ceiling 6.5k→17k up to
   batch ~64). Nearly free extra data; this is *why* games-per-iter barely changes
   iteration time (forward is latency-bound at small batch).
2. **Multiprocessing self-play across the 4 vCPUs.** The CPU bottleneck is pinned to ONE
   core; 3 sit idle. This also fixes CPU/GPU overlap (while one worker's CPU runs,
   another's forward runs), which is what actually keeps the GPU fed.

Expected after both: self-play ~5–6k → **~17k states/s (GPU-bound)**.

**Status (2026-09-22): both done.** #1 = `--games-per-iter 64`. #2 implemented as
`ParallelSelfPlay` in `twenty48/parallel.py` (persistent spawn pool, N workers × 1 core,
own CUDA context each; workers return records, main aggregates into buffer/normalizer;
normalizer frozen during the iteration's self-play). Enable with `--sp-workers N`
(default 1). **Measured: GPU util ~50% → ~94%, power ~13 W → ~30 W with 3 workers**
(≈2× throughput). 3 workers is the sweet spot here (2 already saturate the ~17k ceiling;
more just wastes VRAM/CPU since we're then GPU-bound). #3 (native port) remains not worth
it on this card — see below.

### Capacity math → why a C++/Rust engine port is NOT worth it *on this hardware*

4-core Python CPU ≈ 99 µs ÷ 4 ≈ **~40k states/s** capacity. GPU ceiling ≈ **~17k
states/s.** After the 4-vCPU fix, throughput = min(40k, 17k) = **~17k, GPU-bound** — the
Python CPU already outruns the GPU. A native port would take the CPU to ~400k st/s, all of
it wasted above 17k. Also: the MCTS tree is a Python generator coupled to the batched-
inference driver (yield leaf → receive (policy,value) → resume) — porting it across an FFI
boundary is high-effort/high-risk. **Don't port first; you'd optimize the non-bottleneck.**
(Cheap CPU wins if ever needed: vectorize `encode`, JIT/numba `board.step` — only if
still CPU-bound, which you won't be after 4 cores.)

### ⇒ WHAT CHANGES WITH A BETTER GPU (re-run the benches, then decide)

A stronger GPU raises the ~17k ceiling and adds VRAM. Re-derive the numbers with
`train/gpu_bench.py` (pure-forward ceiling) and `train/profile_split.py` (real split) if
those scratch scripts still exist, else re-create them. Then:

- **The bottleneck can flip back to the CPU.** A native C++/Rust engine port (or numba)
  becomes worthwhile **only once the GPU ceiling exceeds the CPU capacity** — i.e. above
  **~40k states/s on 4 cores** (or ~10k/core if you stay single-threaded). Below that, keep
  Python.
- **Bigger net gets cheaper** (more VRAM + faster forward). Net-capacity/head redesign
  (see below) is the more promising quality lever than more sims.
- **Larger batches pay off** (more VRAM lifts the batch-1024 VRAM wall) — re-find the new
  saturation batch; it will be higher than 64.
- **Higher sims become affordable** — but note sims scale BOTH the forward count and the
  (non-batchable, per-leaf) CPU tree/engine work linearly, so the CPU port question returns.

### Related open quality findings (2026-09-22, not perf)

- 6×6 is **policy/net-capacity limited, not search-limited**: 100→400 sims gave +1.5 %
  score for ~13× cost (sims-sweep). More sims is the wrong lever for 6×6.
- Suspected root cause of weak positional play (no corner-stacking/snaking): the **GAP
  bottleneck** — `adaptive_avg_pool2d` collapses the n×n board to a 128-vector, discarding
  *where* tiles are; the **policy head is a bare `Linear(128→4)`** (value head has one hidden
  layer). Candidate fixes: concat avg+max pool; or a conv-based policy head; give policy head
  a hidden layer. Browser/TS deploy is NO LONGER a constraint (user dropped it) — net may
  grow freely for training quality.

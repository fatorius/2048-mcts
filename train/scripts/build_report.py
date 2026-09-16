"""Constrói o relatório de treino a partir dos metrics.jsonl das runs.

Concatena a linhagem de runs (cada uma retomada da anterior) numa série contínua
de "score x jogos jogados", grava um `metrics.json` (consumido pelo report.html)
e, se matplotlib estiver instalado, um `curve.png` simples.

Uso:
  python scripts/build_report.py                      # usa a linhagem padrão abaixo
  python scripts/build_report.py checkpoints/run_A checkpoints/run_B ...  # explícito

Saída: checkpoints/metrics.json  e  checkpoints/curve.png
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

CKPT = Path(__file__).resolve().parent.parent / "checkpoints"

# Linhagem padrão, em ordem cronológica (cada run retomada da melhor da anterior).
# Acrescente novas runs ao final conforme forem sendo criadas.
DEFAULT_LINEAGE = [
    "run_20260913_193603",  # sims 200
    "run_20260914_011948",  # sims 400
    "run_20260915_095659",  # sims 800
    "run_20260916_084245",  # sims 800 + reward-to-go
]

# Rótulo por run p/ o gráfico (distingue regimes com o mesmo nº de sims).
LABELS = {
    "run_20260913_193603": "200 sims",
    "run_20260914_011948": "400 sims",
    "run_20260915_095659": "800 sims",
    "run_20260916_084245": "800 + reward-to-go",
}

TARGET_REF = 6400  # meta de eval (barra de sucesso); ajuste se quiser.


def load_run(run_dir: Path):
    cfg = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    gpi = int(cfg.get("games_per_iter", 16))
    sims = cfg.get("sims")
    rows = []
    mpath = run_dir / "metrics.jsonl"
    if mpath.exists():
        for line in mpath.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sims, gpi, rows


def build(run_dirs):
    points, phases = [], []
    cum_games = 0
    gi = 0
    for rd in run_dirs:
        rd = Path(rd)
        if not rd.is_absolute():
            rd = (CKPT / rd) if not str(rd).startswith("checkpoints") else (CKPT.parent / rd)
        sims, gpi, rows = load_run(rd)
        if not rows:
            print(f"  aviso: sem métricas em {rd}", file=sys.stderr)
            continue
        lo = cum_games
        for r in rows:
            cum_games += gpi
            gi += 1
            points.append({
                "games": cum_games,
                "iter_global": gi,
                "sims": sims,
                "eval": round(r.get("eval_mean_score", 0), 1),
                "selfplay": round(r.get("selfplay_mean_score", 0), 1),
                "best_tile": r.get("eval_best_tile"),
                "reach2048": r.get("eval_reach_2048", 0),
            })
        phases.append({
            "sims": sims, "lo": lo, "hi": cum_games, "run": rd.name,
            "label": LABELS.get(rd.name, f"{sims} sims"),
        })

    peak = max((p["eval"] for p in points), default=0)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "peak_ref": round(peak),
        "target_ref": TARGET_REF,
        "total_games": cum_games,
        "total_iters": gi,
        "phases": phases,
        "points": points,
    }
    out_json = CKPT / "metrics.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"gravado {len(points)} pontos ({cum_games} jogos) -> {out_json}")
    return report


def render_png(report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib não instalado — pulando curve.png "
              "(instale com: pip install matplotlib)", file=sys.stderr)
        return

    pts = report["points"]
    if not pts:
        return
    x = [p["games"] for p in pts]
    ev = [p["eval"] for p in pts]
    sp = [p["selfplay"] for p in pts]

    # média móvel centrada (janela 7) do eval
    W = 3
    sm = []
    for i in range(len(ev)):
        lo, hi = max(0, i - W), min(len(ev), i + W + 1)
        sm.append(sum(ev[lo:hi]) / (hi - lo))

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    # faixas de fase
    band_colors = {200: "#eef0f3", 400: "#e3e6ea", 800: "#f6e4d6"}
    for ph in report["phases"]:
        ax.axvspan(ph["lo"], ph["hi"], color=band_colors.get(ph["sims"], "#eeeeee"), zorder=0)
        ax.text(ph["lo"] + 6, ax.get_ylim()[1], f'{ph["sims"]} sims',
                fontsize=8, va="top", ha="left", color="#8a8f98")

    ax.scatter(x, ev, s=9, color="#c9631f", alpha=0.28, zorder=2, label="eval (bruto)")
    ax.plot(x, sm, color="#c9631f", lw=2.2, zorder=4, label="eval (tendência 7pt)")
    ax.plot(x, sp, color="#3f8a80", lw=1.2, alpha=0.55, zorder=3, label="self-play")
    ax.axhline(report["peak_ref"], ls="--", lw=1.2, color="#b0472e", zorder=1,
               label=f'pico {report["peak_ref"]}')
    ax.axhline(report["target_ref"], ls="--", lw=1.2, color="#7b8290", zorder=1,
               label=f'meta {report["target_ref"]}')

    ax.set_xlabel("jogos de self-play acumulados")
    ax.set_ylabel("eval score (16 partidas gulosas @ 400 sims)")
    ax.set_title("2048 · curva de aprendizado 4×4")
    ax.set_ylim(0, max(8000, max(ev) * 1.1))
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out_png = CKPT / "curve.png"
    fig.savefig(out_png)
    print(f"gravado -> {out_png}")


def main():
    runs = sys.argv[1:] or DEFAULT_LINEAGE
    report = build(runs)
    render_png(report)


if __name__ == "__main__":
    main()

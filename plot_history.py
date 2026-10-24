"""Curvas de entrenamiento a partir de history.jsonl (una línea por evaluación).

  python plot_history.py work/runs/lora_*_e8 --out curves.png
Paneles: pérdida (train y val ponderada), neto en val.sel, AUC en val (global y en eventos donde el tracker falló).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="carpetas de runs con history.jsonl")
    ap.add_argument("--out", default="curves.png")
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("falta matplotlib: pip install matplotlib")

    fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
    for r in args.runs:
        f = Path(r) / "history.jsonl"
        if not f.exists():
            print(f"[aviso] sin {f}")
            continue
        h = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        x = [e["step"] for e in h]
        name = Path(r).name.replace("lora_", "")
        ax[0].plot(x, [e["train_loss"] for e in h], "--", label=f"{name} train")
        ax[0].plot(x, [e["val_bce_weighted"] or e["val_bce"] for e in h], "-o", ms=3, label=f"{name} val")
        ax[1].plot(x, [e["net"] for e in h], "-o", ms=3, label=name)
        ax[2].plot(x, [e["val_auc"] for e in h], "-o", ms=3, label=name)
        ax[3].plot(x, [e["val_auc_trk_err"] for e in h], "-o", ms=3, label=name)
        b = max(h, key=lambda e: e["net"])
        print(f"{name}: evaluaciones={len(h)} mejor neto={b['net']} (ep {b['ep']} it {b['it']}, beta={b['beta']}, "
              f"tau={b['tau']}, arreglados={b['fixed']}, rotos={b['broken']})")
    for a, t in zip(ax, ("pérdida (train / val ponderada)", "neto en val.sel (arreglados - rotos)",
                         "AUC por par en val", "AUC en eventos con error del tracker")):
        a.set_title(t); a.set_xlabel("paso"); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("->", args.out)


if __name__ == "__main__":
    main()

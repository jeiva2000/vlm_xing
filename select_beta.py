"""Elige beta y tau con la validación de SELECCIÓN (val.sel) y los guarda como selection.json.

Se puntúa el adaptador sobre val.sel (evaluate.py --adapter ... --events *.sel.jsonl) y aquí se busca el
(beta, tau) de mayor neto (arreglados - rotos) de la rejilla de assoc.sweep. En empate gana el beta más
alto (más cercano al tracker). Así el beta/tau que usan apply_vlm.py y visualize_cases.py no se eligen
sobre test.

  python select_beta.py --events events/val/events_bytetrack.sel.jsonl \
      --scores runs/sel_x/scores.jsonl --out runs/sel_x/selection.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from assoc import sweep
from common import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    events = read_jsonl(Path(args.events))
    sw = sweep(events, read_jsonl(Path(args.scores)))
    best = max(sw, key=lambda r: (r["net"], r["beta"]))
    Path(args.out).write_text(json.dumps(best, indent=1))
    print(f"{'beta':>6} {'tau':>5} {'fixed':>6} {'broken':>7} {'net':>5}")
    for r in sw:
        print(f"{r['beta']:>6g} {r['tau']:>5g} {r['fixed']:>6} {r['broken']:>7} {r['net']:>5}"
              + ("  <-" if r is best else ""))
    print(f"seleccionado: beta={best['beta']} tau={best['tau']} neto={best['net']} -> {args.out}")


if __name__ == "__main__":
    main()

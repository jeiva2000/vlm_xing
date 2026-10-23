"""Evalúa con TrackEval (HOTA, DetA, AssA, MOTA, IDF1, IDSW) varias variantes de salida de tracking.

Cada variante es una carpeta con <seq>.txt en formato MOT. Típicamente: el tracker solo y el
tracker + VLM (apply_vlm.py). Ejemplo:
  python eval_tracks.py --gt_root $DT/val --events events/val/events_bytetrack.test.jsonl \
      --variant bytetrack=events/val/tracks_bytetrack --variant bytetrack+vlm=runs/te/bytetrack/vlm \
      --out runs/te/bytetrack
TrackEval se clona en third_party/ (necesita git e internet la primera vez; python eval_tracks.py --setup).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

TE_DIR = Path(__file__).resolve().parent / "third_party" / "TrackEval"
TE_URL = "https://github.com/JonathonLuiten/TrackEval.git"


def ensure_trackeval() -> None:
    """Clona TrackEval y le quita los alias np.float/np.int/np.bool (rotos en numpy >= 1.24)."""
    if not (TE_DIR / "trackeval").is_dir():
        TE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", TE_URL, str(TE_DIR)], check=True)
    if not (TE_DIR / ".numpy_patched").exists():
        for f in (TE_DIR / "trackeval").rglob("*.py"):
            t = f.read_text()
            n = re.sub(r"\bnp\.(float|int|bool)\b", r"\1", t)
            if n != t:
                f.write_text(n)
        (TE_DIR / ".numpy_patched").write_text("ok\n")
    if str(TE_DIR) not in sys.path:
        sys.path.insert(0, str(TE_DIR))


def evaluate(gt_root: Path, seqs: list[str], variants: dict[str, Path], out: Path) -> dict:
    ensure_trackeval()
    import trackeval

    work = out / "trackeval_data"
    if work.exists():
        shutil.rmtree(work)
    safe = {n: re.sub(r"[^A-Za-z0-9_.-]", "_", n) for n in variants}
    for name, d in variants.items():
        for s in seqs:
            dst = work / "trackers" / safe[name] / "data" / f"{s}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = Path(d) / f"{s}.txt"
            dst.write_text(src.read_text() if src.exists() else "")
    seqmap = work / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")

    ecfg = trackeval.Evaluator.get_default_eval_config()
    ecfg.update(USE_PARALLEL=False, PRINT_CONFIG=False, PRINT_RESULTS=False, PRINT_ONLY_COMBINED=True,
                OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False, TIME_PROGRESS=False)
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update(GT_FOLDER=str(gt_root), TRACKERS_FOLDER=str(work / "trackers"), SEQMAP_FILE=str(seqmap),
                SKIP_SPLIT_FOL=True, TRACKER_SUB_FOLDER="data", TRACKERS_TO_EVAL=list(safe.values()),
                DO_PREPROC=False, PRINT_CONFIG=False, OUTPUT_FOLDER=str(work / "out"))
    mcfg = {"PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(mcfg), trackeval.metrics.CLEAR(mcfg), trackeval.metrics.Identity(mcfg)]
    res, _ = trackeval.Evaluator(ecfg).evaluate([trackeval.datasets.MotChallenge2DBox(dcfg)], metrics)
    res = res["MotChallenge2DBox"]

    table = {}
    for name in variants:
        r = res[safe[name]]["COMBINED_SEQ"]["pedestrian"]
        table[name] = {
            "HOTA": 100 * float(np.mean(r["HOTA"]["HOTA"])), "DetA": 100 * float(np.mean(r["HOTA"]["DetA"])),
            "AssA": 100 * float(np.mean(r["HOTA"]["AssA"])), "MOTA": 100 * float(r["CLEAR"]["MOTA"]),
            "IDF1": 100 * float(r["Identity"]["IDF1"]), "IDSW": int(r["CLEAR"]["IDSW"]),
        }
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true", help="solo clonar/parchear TrackEval y salir")
    ap.add_argument("--gt_root", help="carpeta del split con <seq>/gt/gt.txt y seqinfo.ini (p.ej. DanceTrack/val)")
    ap.add_argument("--events", help="JSONL de eventos; se evalúan las secuencias que contiene")
    ap.add_argument("--seqs", nargs="*", help="secuencias a evaluar (alternativa a --events)")
    ap.add_argument("--variant", action="append", default=[], metavar="NOMBRE=CARPETA")
    ap.add_argument("--out", help="carpeta de salida (metrics_trackeval.json)")
    args = ap.parse_args()
    if args.setup:
        ensure_trackeval()
        print("TrackEval listo en", TE_DIR)
        return
    if not (args.gt_root and args.variant and args.out and (args.events or args.seqs)):
        ap.error("faltan --gt_root, --variant, --out y --events/--seqs")
    if args.events:
        from common import read_jsonl
        seqs = sorted({e["seq"] for e in read_jsonl(Path(args.events))})
    else:
        seqs = sorted(args.seqs)
    variants = dict(v.split("=", 1) for v in args.variant)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = evaluate(Path(args.gt_root), seqs, {k: Path(v) for k, v in variants.items()}, out)
    (out / "metrics_trackeval.json").write_text(json.dumps({"seqs": seqs, "metrics": table}, indent=1))
    print(f"secuencias evaluadas: {len(seqs)}")
    print(f"{'variante':<24}{'HOTA':>7}{'DetA':>7}{'AssA':>7}{'MOTA':>7}{'IDF1':>7}{'IDSW':>7}")
    for n, m in table.items():
        print(f"{n:<24}{m['HOTA']:>7.2f}{m['DetA']:>7.2f}{m['AssA']:>7.2f}{m['MOTA']:>7.2f}{m['IDF1']:>7.2f}{m['IDSW']:>7d}")


if __name__ == "__main__":
    main()

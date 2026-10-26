"""Aplica las decisiones del VLM sobre las trayectorias del tracker y escribe MOT por secuencia.

Es un post-proceso: el tracker corre igual que al minar eventos (mismos IDs), y para cada evento
cuya decisión final (assoc.decide con beta/tau) asigna a la identidad de una memoria otro candidato,
se re-etiqueta ese candidato con el ID de salida de la memoria DESDE EL FRAME DE LA CONSULTA en
adelante (los frames del cruce quedan con el ID del tracker). Los eventos se aplican en orden
temporal, componiendo los remapeos. Si dos tracks acabaran con el mismo ID de salida, el que no
fue asignado recibe un ID nuevo.

Ejemplo:
  python apply_vlm.py --events events/val/events_bytetrack.test.jsonl \
      --scores runs/final_x_bytetrack/scores.jsonl --tracks_dir events/val/tracks_bytetrack \
      --selection runs/lora/best/selection.json --out_dir runs/trackeval/bytetrack/vlm
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

from assoc import decide, group_scores
from common import read_jsonl


def read_tracks(path: Path) -> dict[int, list]:
    by = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            p = line.split(",")
            x, y, w, h = map(float, p[2:6])  # el MOT guarda x,y,w,h
            by[int(p[0])].append((int(p[1]), x, y, x + w, y + h))
    return by


def apply_event(ev: dict, final: list, remap: dict, fresh) -> int:
    """Actualiza remap (id del tracker -> id de salida). Devuelve cuántos tracks cambian de ID."""
    cur = lambda t: remap.get(t, t)  # noqa: E731
    new: dict[int, int] = {}
    for m, fin in zip(ev["memories"], final):
        if fin is not None and fin not in new:
            new[fin] = cur(m["tid"])  # el candidato elegido hereda la identidad (ID de salida) de la memoria
    targets = set(new.values())
    for c in ev["candidates"]:
        t = c["tid"]
        if t not in new and cur(t) in targets:  # evitar dos tracks con el mismo ID de salida
            new[t] = next(fresh)
    changed = sum(cur(t) != o for t, o in new.items())
    remap.update(new)
    return changed


def remap_sequence(tracks: dict[int, list], evs: list[tuple[int, dict, list]]):
    """evs: [(query_frame, evento, decisión)] -> (filas (frame, id, x, y, w, h), n_cambios, n_choques)."""
    remap: dict[int, int] = {}
    fresh = itertools.count(1_000_000)
    evs = sorted(evs, key=lambda x: x[0])
    ptr, changed, clashes, rows = 0, 0, 0, []
    for f in sorted(tracks):
        while ptr < len(evs) and evs[ptr][0] <= f:
            changed += apply_event(evs[ptr][1], evs[ptr][2], remap, fresh)
            ptr += 1
        seen = set()
        # los tracks re-etiquetados explícitamente conservan su ID; el no asignado cede
        for tid, x1, y1, x2, y2 in sorted(tracks[f], key=lambda r: r[0] not in remap):
            out = remap.get(tid, tid)
            if out in seen:
                clashes += 1
                out = remap[tid] = next(fresh)  # ID nuevo ESTABLE (no uno por frame)
            seen.add(out)
            rows.append((f, out, x1, y1, x2, y2))
    return rows, changed, clashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="JSONL de eventos (el MISMO que se puntuó)")
    ap.add_argument("--scores", required=True, help="scores.jsonl de evaluate.py sobre esos eventos")
    ap.add_argument("--tracks_dir", required=True, help="carpeta con <seq>.txt crudos del tracker (mine_events)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--selection", help="selection.json del mejor checkpoint (trae beta y tau)")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    args = ap.parse_args()

    beta, tau = args.beta, args.tau
    if args.selection:
        sel = json.loads(Path(args.selection).read_text())
        beta = sel["beta"] if beta is None else beta
        tau = sel["tau"] if tau is None else tau
    if beta is None or tau is None:
        ap.error("define --selection o --beta y --tau")

    events = read_jsonl(Path(args.events))
    g = group_scores(read_jsonl(Path(args.scores)))
    by_seq = defaultdict(list)
    n_dec = n_diff = 0
    for ei, ev in enumerate(events):
        sc = g.get(ei)
        if not sc:
            continue
        final = decide(ev, sc, beta, tau)
        n_dec += 1
        n_diff += any(fin != m["tracker_choice"] for m, fin in zip(ev["memories"], final))
        by_seq[ev["seq"]].append((ev["query_frame"], ev, final))
    print(f"beta={beta} tau={tau}  eventos decididos={n_dec}  donde el VLM cambia al tracker={n_diff}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tot_ch = tot_cl = 0
    for seq in sorted({e["seq"] for e in events}):
        tracks = read_tracks(Path(args.tracks_dir) / f"{seq}.txt")
        ids = {r[0] for rows_f in tracks.values() for r in rows_f}
        ev_ids = {t["tid"] for _, e, _ in by_seq[seq] for t in e["memories"] + e["candidates"]}
        if ev_ids and not ev_ids & ids:
            raise SystemExit(f"{seq}: ningún ID de los eventos existe en {args.tracks_dir} (IDs desfasados: "
                             f"eventos {min(ev_ids)}-{max(ev_ids)}, pistas {min(ids)}-{max(ids)}). "
                             "Las pistas deben generarse con el mismo orden/conjunto de secuencias que el minado.")
        rows, ch, cl = remap_sequence(tracks, by_seq[seq])
        tot_ch += ch; tot_cl += cl
        with open(out / f"{seq}.txt", "w") as fh:
            for f, t, x1, y1, x2, y2 in rows:
                fh.write(f"{f},{t},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1\n")
    print(f"tracks re-etiquetados={tot_ch}  choques de ID resueltos={tot_cl} -> {out}")


if __name__ == "__main__":
    main()

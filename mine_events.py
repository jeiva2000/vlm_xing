"""Paso 1: ejecutar trackers (ByteTrack / BoT-SORT / OC-SORT) y extraer eventos de cruce.

Reglas para que el dataset se parezca al caso online:
  * El DISPARADOR no usa GT: solo solapamiento entre tracks activos (IoU > trig_iou).
  * La MEMORIA se congela en el frame en que salta el disparador (frames < trigger).
    Si el tracker ya se había equivocado antes, la memoria queda contaminada: se
    guarda igual (purity < 1) porque eso es lo que ocurrirá online.
  * La CONSULTA se hace cuando los tracks llevan sep_frames sin solaparse.
  * Se guardan TODOS los eventos, también los que el tracker resolvió bien, para
    poder medir arreglados vs. rotos.
  * El GT solo se usa para etiquetar (y para medir el recall del disparador).

Uso (por defecto detecta con YOLOX-X DanceTrack; descarga pesos y cachea detecciones):
  python mine_events.py --data_root /data/DanceTrack/val --det_dir dets/yolox_dancetrack/val \
      --trackers bytetrack botsort ocsort --out_dir events/val --fp16
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from common import iou_matrix, list_sequences, match_to_gt, read_mot_txt, read_seqinfo, frame_path, write_jsonl
from tracker_adapter import make_tracker, step


class Event:
    _next = 0

    def __init__(self, start: int):
        self.id = Event._next
        Event._next += 1
        self.start = start
        self.last_overlap = start
        self.tracks: set[int] = set()
        self.new_tracks: set[int] = set()
        self.mem: dict[int, list] = {}
        self.region = None

    def add_track(self, tid: int, hist: dict):
        if tid not in self.tracks:
            self.tracks.add(tid)
            self.mem[tid] = list(hist.get(tid, []))  # congelada: solo frames anteriores

    def absorb(self, other: "Event"):
        self.start = min(self.start, other.start)
        self.last_overlap = max(self.last_overlap, other.last_overlap)
        self.tracks |= other.tracks
        self.new_tracks |= other.new_tracks
        for k, v in other.mem.items():
            self.mem.setdefault(k, v)


def _expand(box, s=1.5):
    cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
    return np.array([cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2])


def _major(gts):
    valid = [g for g in gts if g >= 0]
    if not valid:
        return -1, 0.0
    g, n = Counter(valid).most_common(1)[0]
    return int(g), n / len(gts)


def build_record(ev: Event, f: int, cur: dict, cur_gt: dict, seq_name: str, tracker: str,
                 min_mem: int, timeout: bool):
    memories = []
    for tid in sorted(ev.tracks):
        m = ev.mem.get(tid, [])
        if len(m) < min_mem:
            continue
        gts = [e[2] for e in m]
        major, purity = _major(gts)
        memories.append({
            "tid": int(tid), "frames": [int(e[0]) for e in m],
            "boxes": [list(map(float, e[1])) for e in m],
            "gt_ids": [int(g) for g in gts], "gt_major": major, "purity": purity,
        })
    cands = [{"tid": int(t), "box": list(map(float, cur[t])), "gt_id": int(cur_gt[t]),
              "is_new": t in ev.new_tracks}
             for t in sorted(ev.tracks | ev.new_tracks) if t in cur]
    if not memories or not cands:
        return None
    tracker_ok = []
    for m in memories:
        dec = next((c["tid"] for c in cands if c["tid"] == m["tid"]), None)
        truth = next((c["tid"] for c in cands if m["gt_major"] >= 0 and c["gt_id"] == m["gt_major"]), None)
        m["tracker_choice"], m["truth_choice"] = dec, truth
        tracker_ok.append(dec == truth)
    return {
        "seq": seq_name, "tracker": tracker, "event_id": ev.id,
        "start": ev.start, "query_frame": f, "timeout": timeout,
        "memories": memories, "candidates": cands,
        "tracker_error": not all(tracker_ok),
    }


def gt_id_switches(assign: dict[int, list[tuple[int, int]]]) -> list[tuple[int, int, int, int]]:
    """assign[gt_id] = [(frame, tid)...] -> lista de (frame, gt_id, tid_prev, tid_new)."""
    sw = []
    for g, seq in assign.items():
        seq.sort()
        for (f0, t0), (f1, t1) in zip(seq, seq[1:]):
            if t0 != t1:
                sw.append((f1, g, t0, t1))
    return sw


def run_sequence(seq_dir: Path, tracker_name: str, args, tracker_kwargs: dict):
    info = read_seqinfo(seq_dir)
    gt = read_mot_txt(seq_dir / "gt" / "gt.txt", is_gt=True)
    if args.det_source == "gt":
        rng = np.random.default_rng(0)
        dets = {}
        for f, g in gt.items():
            d = g[:, 1:6].copy()
            d[:, :4] += rng.normal(0, args.gt_jitter, size=(len(d), 4))
            keep = rng.random(len(d)) >= args.gt_drop
            d[:, 4] = 0.9
            dets[f] = d[keep]
    else:
        raw = read_mot_txt(Path(args.det_dir) / f"{seq_dir.name}.txt", is_gt=False)
        dets = {f: d[:, 1:6] for f, d in raw.items()}

    trk = make_tracker(tracker_name, **tracker_kwargs)
    hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=args.mem_len))
    seen: set[int] = set()
    open_evs: list[Event] = []
    records = []
    gt_assign: dict[int, list] = defaultdict(list)
    event_spans: list[tuple[int, int, set]] = []

    for f in range(1, info["length"] + 1):
        img = cv2.imread(str(frame_path(seq_dir, info, f)))
        d = dets.get(f, np.zeros((0, 5), np.float32))
        d = d[d[:, 4] >= args.min_score] if len(d) else d
        ids, boxes = step(trk, d, img)
        gts = match_to_gt(boxes, gt.get(f), thr=args.gt_iou)
        cur = {int(t): b for t, b in zip(ids, boxes)}
        cur_gt = {int(t): int(g) for t, g in zip(ids, gts)}
        for t, g in cur_gt.items():
            if g >= 0:
                gt_assign[g].append((f, t))
        born = [t for t in cur if t not in seen]
        seen.update(born)

        # --- disparador online (sin GT) ---
        tids = list(cur.keys())
        if len(tids) >= 2:
            iou = iou_matrix(np.stack([cur[t] for t in tids]), np.stack([cur[t] for t in tids]))
            np.fill_diagonal(iou, 0)
            ii, jj = np.where(np.triu(iou) > args.trig_iou)
            for i, j in zip(ii, jj):
                a, b = tids[i], tids[j]
                hits = [e for e in open_evs if a in e.tracks or b in e.tracks]
                if hits:
                    ev = hits[0]
                    for other in hits[1:]:
                        ev.absorb(other)
                        open_evs.remove(other)
                else:
                    ev = Event(f)
                    open_evs.append(ev)
                ev.add_track(a, hist)
                ev.add_track(b, hist)
                ev.last_overlap = f

        # --- tracks nuevos cerca de un evento abierto (posible switch a ID nuevo) ---
        for ev in open_evs:
            present = [cur[t] for t in ev.tracks if t in cur]
            if present:
                ps = np.stack(present)
                ev.region = np.array([ps[:, 0].min(), ps[:, 1].min(), ps[:, 2].max(), ps[:, 3].max()])
            if ev.region is None:
                continue
            reg = _expand(ev.region, args.region_scale)[None]
            for t in born:
                if t not in ev.tracks and iou_matrix(cur[t][None], reg)[0, 0] > 0:
                    ev.new_tracks.add(t)

        # --- cierre de eventos ---
        still = []
        for ev in open_evs:
            sep = f - ev.last_overlap >= args.sep_frames
            timeout = f - ev.start >= args.max_event_len
            if sep or timeout:
                rec = build_record(ev, f, cur, cur_gt, seq_dir.name, tracker_name, args.min_mem, timeout)
                if rec is not None:
                    records.append(rec)
                event_spans.append((ev.start, f, ev.tracks | ev.new_tracks))
            else:
                still.append(ev)
        open_evs = still

        # actualizar historial DESPUÉS del disparador -> memoria congelada sin el frame actual
        for t, b in cur.items():
            hist[t].append((f, b, cur_gt[t]))

    # --- P0: recall del disparador sobre los ID switches reales ---
    sw = gt_id_switches(gt_assign)
    covered = sum(any(s - args.sep_frames <= fs <= e + 1 and (t0 in tr or t1 in tr)
                      for s, e, tr in event_spans) for fs, _, t0, t1 in sw)
    return records, {"switches": len(sw), "covered": covered}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="carpeta con secuencias (p.ej. DanceTrack/val)")
    ap.add_argument("--det_source", choices=["yolox", "file", "gt"], default="yolox",
                    help="yolox: genera (si faltan) y usa detecciones de YOLOX-X DanceTrack")
    ap.add_argument("--det_dir", help="detecciones MOT por secuencia: <det_dir>/<seq>.txt")
    ap.add_argument("--yolox_weights", default="models/yolox_x_dancetrack.pt")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--det_batch", type=int, default=4)
    ap.add_argument("--gt_jitter", type=float, default=3.0, help="solo con det_source=gt (depuración)")
    ap.add_argument("--gt_drop", type=float, default=0.05)
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort", "ocsort"])
    ap.add_argument("--reid", action="store_true",
                    help="BoT-SORT con ReID (descarga los pesos si faltan; sin la opción va sin apariencia)")
    ap.add_argument("--reid_weights", default="models/osnet_x0_25_msmt17.pt")
    ap.add_argument("--tracker_kwargs", default="{}", help="JSON con overrides para make_tracker")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seqs", nargs="*", help="subconjunto de secuencias")
    ap.add_argument("--min_score", type=float, default=0.1)
    ap.add_argument("--gt_iou", type=float, default=0.5)
    ap.add_argument("--trig_iou", type=float, default=0.2)
    ap.add_argument("--sep_frames", type=int, default=3)
    ap.add_argument("--max_event_len", type=int, default=150)
    ap.add_argument("--mem_len", type=int, default=30, help="frames guardados por memoria")
    ap.add_argument("--min_mem", type=int, default=3)
    ap.add_argument("--region_scale", type=float, default=1.5)
    args = ap.parse_args()
    if args.det_source in ("file", "yolox") and not args.det_dir:
        ap.error("--det_dir es obligatorio con det_source=file|yolox")

    root = Path(args.data_root)
    seqs = [s for s in list_sequences(root) if not args.seqs or s.name in args.seqs]

    if args.det_source == "yolox":  # genera solo las detecciones que falten
        missing = [s for s in seqs if not (Path(args.det_dir) / f"{s.name}.txt").exists()]
        if missing:
            from detect_yolox import YoloXDanceTrack, detect_sequence
            det = YoloXDanceTrack(args.yolox_weights, fp16=args.fp16)
            for s in missing:
                detect_sequence(det, s, Path(args.det_dir) / f"{s.name}.txt", args.det_batch)
            del det
    tk = json.loads(args.tracker_kwargs)
    tk.update(reid=args.reid, reid_weights=args.reid_weights)  # solo los usa BoT-SORT
    for tname in args.trackers:
        all_recs, n_sw, n_cov = [], 0, 0
        for s in seqs:
            recs, st = run_sequence(s, tname, args, dict(tk))
            for r in recs:
                r["seq_dir"] = str(s)
            all_recs += recs
            n_sw += st["switches"]
            n_cov += st["covered"]
            print(f"[{tname}] {s.name}: eventos={len(recs)} switches={st['switches']} cubiertos={st['covered']}")
        out = Path(args.out_dir) / f"events_{tname}.jsonl"
        write_jsonl(out, all_recs)
        n_err = sum(r["tracker_error"] for r in all_recs)
        print(f"[{tname}] TOTAL eventos={len(all_recs)} con_error={n_err} "
              f"recall_disparador={n_cov}/{n_sw} -> {out}")


if __name__ == "__main__":
    main()

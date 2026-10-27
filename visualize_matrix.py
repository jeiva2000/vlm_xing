"""Asociaciones en formato consulta vs banco: una imagen por EVENTO con la matriz completa.

  filas     = BANCO: cada memoria (identidad previa al cruce), con sus últimos frames de referencia
  columnas  = CONSULTA: cada candidato del frame de la consulta (recorte), con su tid y su GT
  celdas    = score del VLM (logit_yes - logit_no); color rojo (no es) -> verde (es)
              borde verde grueso = candidato correcto según el GT
              T = elección del tracker, V = elección final del VLM (con beta/tau), +b = bonus beta
  esquina   = frame de la consulta con las cajas de los candidatos numeradas

A la derecha de cada fila: veredicto del tracker y del VLM (OK / MAL).
Los eventos se agrupan por su peor caso: broken > wrong > fixed > correct.

Ejemplos:
  python visualize_matrix.py --events events/val/events_botsort.test.jsonl \
      --scores runs/final_x_botsort/scores.jsonl --selection runs/sel_x/selection.json \
      --out runs/matrix/botsort --cats broken wrong
  python visualize_matrix.py ... --seq dancetrack0063 --event 120     # un evento concreto
"""
from __future__ import annotations

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from assoc import decide, group_scores
from common import FrameCache, frame_path, read_jsonl, read_seqinfo, render_view

TW, TH = 130, 160          # recorte de referencia / candidato
CW, CH = 130, 78           # celda de score
GAP, BANNER = 10, 64
PRIORITY = ("broken", "wrong", "fixed", "correct")
CAT_COLOR = {"fixed": (40, 160, 60), "broken": (200, 50, 50), "correct": (70, 110, 200), "wrong": (200, 130, 30)}


def _font(size: int):
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow antiguo
        return ImageFont.load_default()


def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    im = img.copy()
    im.thumbnail((w - 4, h - 4))
    canvas = Image.new("RGB", (w, h), (28, 28, 28))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def score_color(v: float | None) -> tuple[int, int, int]:
    if v is None:
        return (45, 45, 45)
    t = max(-1.0, min(1.0, v / 6.0))
    return (int(70 + 130 * -t), 50, 55) if t < 0 else (60, int(60 + 120 * t), int(60 + 20 * t))


def classify(m: dict, fin) -> str:
    t_ok, f_ok = m["tracker_choice"] == m["truth_choice"], fin == m["truth_choice"]
    return "correct" if t_ok and f_ok else "fixed" if f_ok else "broken" if t_ok else "wrong"


def pick_rows_cols(ev, sc, final, cats, max_rows, max_cols):
    """Filas: primero las memorias con error/cambio. Columnas: truth/tracker/VLM de esas filas + mejores scores."""
    order = sorted(range(len(ev["memories"])), key=lambda i: (cats[i] == "correct", ev["memories"][i]["tid"]))
    rows = sorted(order[:max_rows], key=lambda i: ev["memories"][i]["tid"])
    must, rest = [], []
    for i in rows:
        m = ev["memories"][i]
        must += [m["truth_choice"], m["tracker_choice"], final[i], m["tid"]]
        top = sorted(range(len(ev["candidates"])), key=lambda j: -sc.get((i, j), -1e9))[:2]
        rest += [ev["candidates"][j]["tid"] for j in top]
    tids = [c["tid"] for c in ev["candidates"]]
    chosen = []
    for t in [t for t in must if t is not None] + rest:
        if t in tids and t not in chosen:
            chosen.append(t)
    cols = sorted((j for j, c in enumerate(ev["candidates"]) if c["tid"] in chosen[:max_cols]),
                  key=lambda j: ev["candidates"][j]["tid"])
    return rows, cols


def render_event(ei, ev, sc, final, cats, cache, n_refs, stride, max_rows, max_cols, beta, tau):
    seq_dir = Path(ev["seq_dir"])
    info = read_seqinfo(seq_dir)
    rows, cols = pick_rows_cols(ev, sc, final, cats, max_rows, max_cols)
    cands = ev["candidates"]
    worst = next(c for c in PRIORITY if c in cats)

    def crop(frame, box):
        return render_view(cache.get(frame_path(seq_dir, info, frame)), box, "crop", ctx=2.5, crop_size=TW * 2)[0]

    corner_w, ver_w, rh = n_refs * TW, 150, max(TH, CH)
    W = GAP + corner_w + GAP + len(cols) * CW + GAP + ver_w + GAP
    H = BANNER + TH + GAP + len(rows) * (rh + 4) + 2 * GAP
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    f1, f2, f3, fs = _font(22), _font(16), _font(13), _font(24)
    d.rectangle([0, 0, W, BANNER - 8], fill=CAT_COLOR[worst])
    d.text((10, 6), f"[{worst.upper()}] {ev['seq']}  evento {ei}  consulta f={ev['query_frame']}  "
                    f"tracker={ev['tracker']}", font=f1, fill="white")
    last = [m["frames"][-1] for m in ev["memories"]]
    d.text((10, 36), f"beta={beta:g} tau={tau:g} | filas = BANCO (memoria congelada, azul = frames antes de la consulta)  "
                     f"columnas = CONSULTA (f={ev['query_frame']}, {ev['query_frame'] - max(last)}-"
                     f"{ev['query_frame'] - min(last)} fr tras la ultima ref)  T=tracker V=VLM  borde verde=verdad",
           font=f3, fill="white")

    # esquina: frame de la consulta con las cajas numeradas
    frame = Image.fromarray(cache.get(frame_path(seq_dir, info, ev["query_frame"]))).convert("RGB")
    s = min(corner_w / frame.width, TH / frame.height)
    small = frame.resize((max(1, int(frame.width * s)), max(1, int(frame.height * s))))
    ds = ImageDraw.Draw(small)
    for j in cols:
        x1, y1, x2, y2 = [v * s for v in cands[j]["box"]]
        ds.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=1)
        ds.text((x1 + 1, y1 + 1), str(cands[j]["tid"]), font=f3, fill=(255, 255, 0))
    canvas.paste(small, (GAP, BANNER))
    d.text((GAP, BANNER + small.height + 1), "frame consulta", font=f3, fill=(170, 170, 170))

    # cabecera de columnas (candidatos)
    for k, j in enumerate(cols):
        c = cands[j]
        x = GAP + corner_w + GAP + k * CW
        canvas.paste(_fit(crop(ev["query_frame"], c["box"]), CW, TH), (x, BANNER))
        d.text((x + 3, BANNER + 2), f"tid {c['tid']}", font=f2, fill=(255, 255, 255))
        d.text((x + 3, BANNER + TH - 18), f"gt {c['gt_id']}  f={ev['query_frame']}", font=f3, fill=(255, 220, 90))

    # filas (memorias)
    y = BANNER + TH + GAP
    for i in rows:
        m, cat = ev["memories"][i], cats[i]
        idxs = list(range(len(m["frames"]) - 1, -1, -stride))[:n_refs][::-1]
        for q, k in enumerate(idxs):
            canvas.paste(_fit(crop(m["frames"][k], m["boxes"][k]), TW, rh), (GAP + q * TW, y))
            fr = m["frames"][k]
            d.text((GAP + q * TW + 3, y + 2), f"f={fr}", font=f3, fill=(255, 255, 255))
            d.text((GAP + q * TW + 3, y + 16), f"-{ev['query_frame'] - fr} fr", font=f3, fill=(150, 200, 255))
        d.text((GAP + 3, y + rh - 36), f"mem tid {m['tid']}", font=f2, fill=(255, 255, 255))
        d.text((GAP + 3, y + rh - 18), f"gt {m['gt_major']}  pureza {m['purity']:.2f}  hist {len(m['frames'])} fr",
               font=f3, fill=(255, 220, 90))
        cy = y + (rh - CH) // 2
        for k, j in enumerate(cols):
            c = cands[j]
            x = GAP + corner_w + GAP + k * CW
            v = sc.get((i, j))
            d.rectangle([x, cy, x + CW - 2, cy + CH - 2], fill=score_color(v))
            d.text((x + 8, cy + 8), "-" if v is None else f"{v:+.2f}", font=fs, fill="white")
            if c["tid"] == m["tid"]:
                d.text((x + 8, cy + CH - 22), "+b (mismo tid)", font=f3, fill=(255, 255, 255))
            if c["tid"] == m["truth_choice"]:
                d.rectangle([x, cy, x + CW - 2, cy + CH - 2], outline=(70, 230, 90), width=5)
            tag = ("T" if c["tid"] == m["tracker_choice"] else "") + ("V" if c["tid"] == final[i] else "")
            if tag:
                d.rectangle([x + CW - 40, cy + 4, x + CW - 6, cy + 30], fill=(0, 0, 0))
                d.text((x + CW - 36, cy + 5), tag, font=f2, fill=(255, 220, 90))
        xv = GAP + corner_w + GAP + len(cols) * CW + GAP
        d.rectangle([xv, y, xv + ver_w - 4, y + rh - 4], fill=CAT_COLOR[cat])
        d.text((xv + 8, y + 6), cat.upper(), font=f2, fill="white")
        tk, vl = m["tracker_choice"] == m["truth_choice"], final[i] == m["truth_choice"]
        d.text((xv + 8, y + 32), f"tracker: {'OK' if tk else 'MAL'} ({m['tracker_choice']})", font=f3, fill="white")
        d.text((xv + 8, y + 52), f"VLM: {'OK' if vl else 'MAL'} ({final[i]})", font=f3, fill="white")
        d.text((xv + 8, y + 72), f"verdad: {m['truth_choice']}", font=f3, fill="white")
        y += rh + 4
    return canvas, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--cats", nargs="+", default=["broken", "wrong", "fixed"], choices=PRIORITY,
                    help="se muestran eventos cuyo peor caso esté en estas categorías")
    ap.add_argument("--seq", help="solo esta secuencia")
    ap.add_argument("--event", type=int, help="solo este índice de evento (dentro del JSONL)")
    ap.add_argument("--max_per_cat", type=int, default=15)
    ap.add_argument("--max_rows", type=int, default=6)
    ap.add_argument("--max_cols", type=int, default=8)
    ap.add_argument("--n_refs", type=int, default=4, help="frames de referencia por memoria (= n_mem del modelo)")
    ap.add_argument("--mem_stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
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
    finals, cats, by_cat = {}, {}, defaultdict(list)
    for ei, ev in enumerate(events):
        sc = g.get(ei)
        if not sc or (args.seq and ev["seq"] != args.seq) or (args.event is not None and ei != args.event):
            continue
        finals[ei] = decide(ev, sc, beta, tau)
        cats[ei] = [classify(m, f) for m, f in zip(ev["memories"], finals[ei])]
        by_cat[next(c for c in PRIORITY if c in cats[ei])].append(ei)

    out = Path(args.out)
    rng = random.Random(args.seed)
    cache = FrameCache(64)
    index, summary = defaultdict(list), {c: len(by_cat[c]) for c in PRIORITY}
    for cat in PRIORITY:
        if cat not in args.cats and args.event is None:
            continue
        pool = by_cat[cat]
        for ei in sorted(rng.sample(pool, min(args.max_per_cat, len(pool)))):
            ev = events[ei]
            try:
                im, _ = render_event(ei, ev, g[ei], finals[ei], cats[ei], cache, args.n_refs, args.mem_stride,
                                     args.max_rows, args.max_cols, beta, tau)
            except FileNotFoundError as e:
                print(f"[aviso] sin frame {e}; se omite el evento {ei}")
                continue
            rel = f"{cat}/{ev['seq']}_e{ei}.jpg"
            (out / cat).mkdir(parents=True, exist_ok=True)
            im.save(out / rel, quality=88)
            index[cat].append(rel)

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"beta": beta, "tau": tau, "eventos_por_peor_caso": summary}, indent=1))
    parts = ["<html><meta charset='utf-8'><body style='background:#111;color:#eee;font-family:sans-serif'>"
             f"<h2>Consulta vs banco (beta={beta:g}, tau={tau:g})</h2>"]
    for cat in PRIORITY:
        if index[cat]:
            parts.append(f"<h3>{cat}: {summary[cat]} eventos (se muestran {len(index[cat])})</h3>")
            parts += [f"<div><img src='{html.escape(r)}' style='max-width:100%;margin:6px 0'></div>" for r in index[cat]]
    (out / "index.html").write_text("\n".join(parts))
    print(json.dumps(summary), "->", out / "index.html")


if __name__ == "__main__":
    main()

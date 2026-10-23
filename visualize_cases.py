"""Imágenes de casos de validación: dónde el VLM acierta y dónde falla.

Por cada memoria de cada evento se compara la decisión del tracker, la del VLM (assoc.decide con
beta/tau) y la verdad (GT), y se clasifica en:
  fixed    el tracker fallaba y el VLM lo arregla
  broken   el tracker acertaba y el VLM lo rompe
  correct  ambos aciertan
  wrong    ambos fallan
Cada caso es una imagen: frames de referencia de la memoria (caja roja, de antiguo a reciente) y los
candidatos del frame de la consulta con su score. Borde verde = candidato correcto según el GT.
Etiquetas bajo cada candidato: TRUTH (verdad), TRK (elección del tracker), VLM (elección del VLM).

Ejemplo:
  python visualize_cases.py --events events/val/events_bytetrack.test.jsonl \
      --scores runs/final_x_bytetrack/scores.jsonl --selection runs/lora/best/selection.json \
      --out runs/cases/bytetrack --max_per_cat 12
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

TILE_W, TILE_H, CAP_H, BANNER_H, GAP = 220, 280, 44, 70, 24
CATS = ("fixed", "broken", "correct", "wrong")
CAT_COLOR = {"fixed": (40, 160, 60), "broken": (200, 50, 50), "correct": (70, 110, 200), "wrong": (200, 130, 30)}


def _font(size: int):
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow antiguo
        return ImageFont.load_default()


def _tile(img: Image.Image, border=None) -> Image.Image:
    im = img.copy()
    im.thumbnail((TILE_W - 8, TILE_H - 8))
    canvas = Image.new("RGB", (TILE_W, TILE_H), (28, 28, 28))
    canvas.paste(im, ((TILE_W - im.width) // 2, (TILE_H - im.height) // 2))
    if border:
        ImageDraw.Draw(canvas).rectangle([0, 0, TILE_W - 1, TILE_H - 1], outline=border, width=6)
    return canvas


def mem_indices(n: int, n_mem: int, stride: int) -> list[int]:
    """Mismos frames de memoria que CrossingPairDataset._mem_indices."""
    return list(range(n - 1, -1, -stride))[:n_mem][::-1]


def pick_candidates(ev, sc, mi, tags, k):
    """Máx. k candidatos: siempre truth/tracker/VLM, el resto por score."""
    idx = list(range(len(ev["candidates"])))
    must = [j for j in idx if tags[j]]
    rest = sorted((j for j in idx if not tags[j]), key=lambda j: -sc.get((mi, j), -1e9))
    return sorted((must + rest)[:k])


def render_case(cat, ev, mi, sc, final, cache, n_mem, stride, max_cands, beta, tau) -> Image.Image:
    m = ev["memories"][mi]
    cands = ev["candidates"]
    seq_dir = Path(ev["seq_dir"])
    info = read_seqinfo(seq_dir)
    trk, truth, fin = m["tracker_choice"], m["truth_choice"], final[mi]
    tags = [[n for n, t in (("TRUTH", truth), ("TRK", trk), ("VLM", fin)) if t == c["tid"]] for c in cands]
    show = pick_candidates(ev, sc, mi, tags, max_cands)

    def crop(frame, box):
        img = cache.get(frame_path(seq_dir, info, frame))
        return render_view(img, box, "crop", ctx=2.5, crop_size=TILE_W * 2)[0]

    tiles = []  # (imagen, línea1, línea2, borde)
    for k in mem_indices(len(m["frames"]), n_mem, stride):
        tiles.append((_tile(crop(m["frames"][k], m["boxes"][k])), f"ref f={m['frames'][k]}", "", None))
    n_ref = len(tiles)
    for j in show:
        c = cands[j]
        s = sc.get((mi, j))
        good = m["gt_major"] >= 0 and c["gt_id"] == m["gt_major"]
        tiles.append((_tile(crop(ev["query_frame"], c["box"]), (60, 200, 80) if good else None),
                      f"tid {c['tid']}  s={s:+.2f}" if s is not None else f"tid {c['tid']}",
                      " ".join(tags[j]), None))

    W = len(tiles) * TILE_W + GAP + 2 * GAP
    canvas = Image.new("RGB", (W, BANNER_H + TILE_H + CAP_H), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    f1, f2 = _font(22), _font(16)
    d.rectangle([0, 0, W, BANNER_H - 6], fill=CAT_COLOR[cat])
    d.text((10, 6), f"[{cat.upper()}] {ev['seq']}  consulta f={ev['query_frame']}  tracker={ev['tracker']}  "
                    f"memoria tid={m['tid']} (gt {m['gt_major']}, pureza {m['purity']:.2f})", font=f1, fill="white")
    d.text((10, 38), f"verdad -> tid {truth} | tracker -> tid {trk} | VLM -> tid {fin}   (beta={beta:g}, tau={tau:g})",
           font=f2, fill="white")
    x = GAP
    for i, (t, l1, l2, _) in enumerate(tiles):
        if i == n_ref:
            x += GAP  # separación entre referencias y candidatos
            d.line([(x - GAP // 2, BANNER_H), (x - GAP // 2, BANNER_H + TILE_H + CAP_H)], fill=(120, 120, 120), width=2)
        canvas.paste(t, (x, BANNER_H))
        d.text((x + 4, BANNER_H + TILE_H + 2), l1, font=f2, fill="white")
        if l2:
            d.text((x + 4, BANNER_H + TILE_H + 22), l2, font=f2, fill=(255, 220, 90))
        x += TILE_W
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--scores", required=True, help="scores.jsonl de evaluate.py sobre esos eventos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", help="selection.json del mejor checkpoint (trae beta y tau)")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--max_per_cat", type=int, default=12)
    ap.add_argument("--max_cands", type=int, default=6)
    ap.add_argument("--n_mem", type=int, default=4)
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
    cases = defaultdict(list)  # cat -> [(ei, mi)]
    finals = {}
    for ei, ev in enumerate(events):
        sc = g.get(ei)
        if not sc:
            continue
        finals[ei] = decide(ev, sc, beta, tau)
        for mi, m in enumerate(ev["memories"]):
            t_ok, f_ok = m["tracker_choice"] == m["truth_choice"], finals[ei][mi] == m["truth_choice"]
            cases["correct" if t_ok and f_ok else "fixed" if f_ok else "broken" if t_ok else "wrong"].append((ei, mi))

    out = Path(args.out)
    rng = random.Random(args.seed)
    cache = FrameCache(64)
    summary, index = {}, defaultdict(list)
    for cat in CATS:
        allc = cases[cat]
        summary[cat] = len(allc)
        chosen = rng.sample(allc, min(args.max_per_cat, len(allc)))
        for ei, mi in sorted(chosen):
            ev = events[ei]
            try:
                im = render_case(cat, ev, mi, g[ei], finals[ei], cache, args.n_mem, args.mem_stride,
                                 args.max_cands, beta, tau)
            except FileNotFoundError as e:  # frames no disponibles en este nodo
                print(f"[aviso] sin frame {e}; se omite el caso")
                continue
            rel = f"{cat}/{ev['seq']}_e{ei}_m{mi}.jpg"
            (out / cat).mkdir(parents=True, exist_ok=True)
            im.save(out / rel, quality=88)
            index[cat].append(rel)
    total = sum(summary.values())
    (out / "summary.json").write_text(json.dumps({"beta": beta, "tau": tau, "memorias": total, **summary}, indent=1))

    parts = [f"<html><meta charset='utf-8'><body style='background:#111;color:#eee;font-family:sans-serif'>"
             f"<h2>Casos de validación (beta={beta:g}, tau={tau:g})</h2>"]
    for cat in CATS:
        parts.append(f"<h3>{cat}: {summary[cat]} de {total} memorias (se muestran {len(index[cat])})</h3>")
        parts += [f"<div><img src='{html.escape(r)}' style='max-width:100%;margin:4px 0'></div>" for r in index[cat]]
    (out / "index.html").write_text("\n".join(parts))
    print(json.dumps(summary), "->", out / "index.html")


if __name__ == "__main__":
    main()

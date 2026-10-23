"""Paso 3a: prueba congelada (zero-shot) o evaluación de un adaptador LoRA.

1) Puntúa todos los pares con el VLM y guarda scores.jsonl (caro, se hace una vez).
2) Calcula métricas por par y por evento con barrido de beta/tau (barato).

Ejemplo (varias GPUs: torchrun --standalone --nproc_per_node N evaluate.py ...):
  python evaluate.py --model qwen3vl-8b --events events/val/events_bytetrack.jsonl \
      --view_mode frame --n_mem 4 --out runs/zs_8b_frame
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from assoc import pair_metrics, sweep
from common import read_jsonl, write_jsonl
from dataset import CrossingPairDataset, VLMCollator
from dist_utils import barrier, cleanup, device_map, gather_rows, init_dist


def add_data_args(ap):
    ap.add_argument("--view_mode", choices=["frame", "crop", "both"], default="frame")
    ap.add_argument("--n_mem", type=int, default=4)
    ap.add_argument("--mem_stride", type=int, default=3)
    ap.add_argument("--max_side", type=int, default=640)
    ap.add_argument("--crop_size", type=int, default=336)
    ap.add_argument("--add_traj_text", action="store_true")
    ap.add_argument("--max_pixels", type=int, default=None, help="límite de píxeles por imagen del processor")


def add_model_args(ap):
    ap.add_argument("--model", default="qwen3vl-8b", help="alias de MODEL_VARIANTS o ID de HF")
    ap.add_argument("--dtype", default="bfloat16", help="float16 en la RTX 2080")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--no_thinking", action="store_true", help="pasa enable_thinking=False (Qwen3.5)")


@torch.no_grad()
def score_dataset(verifier, ds, collator, batch_size, num_workers=4, limit=None, rank=0, world=1):
    """Con world > 1 cada proceso puntúa su porción de pares; hay que juntar las filas con gather_rows."""
    verifier.model.eval()
    if world > 1:
        ds = Subset(ds, range(rank, len(ds), world))
        limit = -(-limit // world) if limit else None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers)
    rows = []
    for bi, batch in enumerate(dl):
        s = verifier.scores(batch).cpu().tolist()
        for m, v, y in zip(batch["meta"], s, batch["labels_bin"].tolist()):
            rows.append({**m, "score": v, "label": int(y)})
        if limit and len(rows) >= limit:
            break
        if bi % 20 == 0 and rank == 0:
            print(f"  {len(rows)}/{len(ds)} pares")
    return rows


def report(events, rows, out_dir: Path, tag: str = ""):
    pm = pair_metrics(rows)
    sw = sweep(events, rows)
    best = max(sw, key=lambda r: r["net"])
    (out_dir / f"metrics{tag}.json").write_text(json.dumps({"pair": pm, "sweep": sw, "best": best}, indent=1))
    print("PAR:", json.dumps(pm))
    print("MEJOR (por neto):", json.dumps(best))
    print(f"{'beta':>6} {'tau':>5} {'fixed':>6} {'broken':>7} {'net':>5} {'acc_trk':>8} {'acc_fin':>8}")
    for r in sw:
        print(f"{r['beta']:>6g} {r['tau']:>5g} {r['fixed']:>6} {r['broken']:>7} {r['net']:>5} "
              f"{r['acc_tracker']:>8.3f} {r['acc_final']:>8.3f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None, help="ruta a un adaptador LoRA entrenado")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse_scores", action="store_true", help="solo recalcula métricas")
    add_data_args(ap)
    add_model_args(ap)
    args = ap.parse_args()
    rank, world, local = init_dist()
    main_proc = rank == 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if main_proc:
        (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    ds = CrossingPairDataset(args.events, view_mode=args.view_mode, n_mem=args.n_mem,
                             mem_stride=args.mem_stride, max_side=args.max_side,
                             crop_size=args.crop_size, add_traj_text=args.add_traj_text)
    print(f"eventos={len(ds.events)} pares={len(ds)}")
    sp = out / "scores.jsonl"
    if args.reuse_scores and sp.exists():
        if not main_proc:
            return
        rows = read_jsonl(sp)
    else:
        from verifier import VLMVerifier
        v = VLMVerifier(args.model, args.dtype, load_4bit=args.load_4bit, attn_impl=args.attn_impl,
                        max_pixels=args.max_pixels, device_map=device_map(world, local))
        if args.adapter:
            from peft import PeftModel
            v.model = PeftModel.from_pretrained(v.model, args.adapter)
        coll = VLMCollator(v.processor, {"enable_thinking": False} if args.no_thinking else None)
        rows = gather_rows(score_dataset(v, ds, coll, args.batch_size, limit=args.limit,
                                         rank=rank, world=world), world)
        if main_proc:
            write_jsonl(sp, rows)
    if main_proc:
        report(ds.events, rows, out)
    barrier(world)
    cleanup(world)


if __name__ == "__main__":
    main()

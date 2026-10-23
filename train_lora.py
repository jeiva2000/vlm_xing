"""Paso 3b: fine-tuning (LoRA) del VLM con BCE sobre logit_yes - logit_no.

Época 0 = la prueba congelada sobre validación (línea base). Después se entrena
y se GUARDA SOLO si mejora el neto (arreglados - rotos) en validación, que es una
métrica a nivel de evento con memorias reales del tracker, no la accuracy por par.

Ejemplo (A40/A100):
  python train_lora.py --model qwen3vl-8b \
      --train_events events/train/events_*.jsonl --val_events events/val/events_bytetrack.jsonl \
      --out runs/lora_8b --epochs 2 --batch_size 2 --grad_accum 8
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CrossingPairDataset, VLMCollator
from evaluate import add_data_args, add_model_args, report, score_dataset
from verifier import VLMVerifier, add_lora


def make_ds(files, args, train: bool):
    return CrossingPairDataset(
        files, view_mode=args.view_mode, n_mem=args.n_mem, mem_stride=args.mem_stride,
        max_side=args.max_side, crop_size=args.crop_size, add_traj_text=args.add_traj_text,
        min_purity=args.min_purity_train if train else 0.0,
        drop_timeout=train, error_oversample=args.error_oversample if train else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_events", nargs="+", required=True)
    ap.add_argument("--val_events", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--train_vision", action="store_true", help="LoRA también en el encoder visual")
    ap.add_argument("--full_llm", action="store_true", help="descongelar el LLM entero (solo modelos pequeños)")
    ap.add_argument("--min_purity_train", type=float, default=0.6,
                    help="descarta memorias muy contaminadas (etiqueta ambigua) SOLO en train")
    ap.add_argument("--error_oversample", type=int, default=1)
    ap.add_argument("--pos_weight", type=float, default=None, help="None = automático neg/pos")
    ap.add_argument("--val_limit", type=int, default=None, help="máx. pares de validación por evaluación")
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--skip_frozen_eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    add_data_args(ap)
    add_model_args(ap)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    tr, va = make_ds(args.train_events, args, True), make_ds(args.val_events, args, False)
    overlap = {e["seq"] for e in tr.events} & {e["seq"] for e in va.events}
    assert not overlap, f"secuencias compartidas entre train y val: {sorted(overlap)[:5]}"
    from dataset import pair_label  # contar etiquetas sin renderizar imágenes
    ytr = np.array([pair_label(tr.events[e], m, c) for e, m, c in tr.pairs])
    pos_weight = args.pos_weight or float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"train: eventos={len(tr.events)} pares={len(tr)} pos={ytr.mean():.3f} pos_weight={pos_weight:.2f}")
    print(f"val:   eventos={len(va.events)} pares={len(va)}")

    v = VLMVerifier(args.model, args.dtype, load_4bit=args.load_4bit, attn_impl=args.attn_impl,
                    max_pixels=args.max_pixels)
    coll = VLMCollator(v.processor, {"enable_thinking": False} if args.no_thinking else None)

    # ---------------- prueba congelada (época 0) ----------------
    best_net = -math.inf
    if not args.skip_frozen_eval:
        print("== Evaluación congelada (zero-shot) ==")
        rows = score_dataset(v, va, coll, args.batch_size * 2, limit=args.val_limit)
        best_net = report(va.events, rows, out, tag="_frozen")["net"]
        print(f"neto congelado = {best_net}  (el fine-tuning debe superarlo para guardarse)")

    # ---------------- descongelar: LoRA o LLM completo ----------------
    if args.full_llm:
        for n, p in v.model.named_parameters():
            p.requires_grad = not any(k in n for k in ("visual", "vision"))
    else:
        add_lora(v, args.lora_r, args.lora_alpha, train_vision=args.train_vision)
    v.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(v.model, "enable_input_require_grads"):
        v.model.enable_input_require_grads()

    params = [p for p in v.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=coll,
                    num_workers=4, drop_last=True)
    total = args.epochs * len(dl) // args.grad_accum
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * total), total)
    eval_every = max(1, len(dl) // args.evals_per_epoch)

    step = 0
    for ep in range(args.epochs):
        v.model.train()
        run = []
        for it, batch in enumerate(dl):
            loss, _ = v.loss(batch, pos_weight)
            (loss / args.grad_accum).backward()
            run.append(loss.item())
            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"ep {ep} step {step}/{total} loss {np.mean(run[-50:]):.4f}")
            if (it + 1) % eval_every == 0 or it + 1 == len(dl):
                rows = score_dataset(v, va, coll, args.batch_size * 2, limit=args.val_limit)
                res = report(va.events, rows, out, tag=f"_ep{ep}_it{it+1}")
                if res["net"] > best_net:
                    best_net = res["net"]
                    v.model.save_pretrained(out / "best")
                    (out / "best" / "selection.json").write_text(json.dumps(res, indent=1))
                    print(f"** guardado: neto={best_net} (beta={res['beta']}, tau={res['tau']})")
                v.model.train()
    print(f"mejor neto en validación: {best_net}")


if __name__ == "__main__":
    main()

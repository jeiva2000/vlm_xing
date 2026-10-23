"""Paso 3b: fine-tuning (LoRA) del VLM con BCE sobre logit_yes - logit_no.

Época 0 = la prueba congelada sobre validación (línea base). Después se entrena
y se GUARDA SOLO si mejora el neto (arreglados - rotos) en validación, que es una
métrica a nivel de evento con memorias reales del tracker, no la accuracy por par.

Varias GPUs (DDP, un modelo por GPU; grad_accum es POR GPU):
  torchrun --standalone --nproc_per_node 4 train_lora.py --model qwen3vl-8b ... --grad_accum 2

Ejemplo (A40/A100):
  python train_lora.py --model qwen3vl-8b \
      --train_events events/train/events_*.jsonl --val_events events/val/events_bytetrack.jsonl \
      --out runs/lora_8b --epochs 2 --batch_size 2 --grad_accum 8
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CrossingPairDataset, VLMCollator
from dist_utils import barrier, cleanup, device_map, gather_rows, init_dist
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
    ap.add_argument("--resume", action="store_true",
                    help="continúa desde out/last si existe (se guarda en cada evaluación)")
    ap.add_argument("--seed", type=int, default=0)
    add_data_args(ap)
    add_model_args(ap)
    args = ap.parse_args()

    rank, world, local = init_dist()
    main_proc = rank == 0
    log = print if main_proc else (lambda *a, **k: None)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if main_proc:
        (out / "args.json").write_text(json.dumps({**vars(args), "world_size": world}, indent=1))

    tr, va = make_ds(args.train_events, args, True), make_ds(args.val_events, args, False)
    overlap = {e["seq"] for e in tr.events} & {e["seq"] for e in va.events}
    assert not overlap, f"secuencias compartidas entre train y val: {sorted(overlap)[:5]}"
    from dataset import pair_label  # contar etiquetas sin renderizar imágenes
    ytr = np.array([pair_label(tr.events[e], m, c) for e, m, c in tr.pairs])
    pos_weight = args.pos_weight or float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    log(f"train: eventos={len(tr.events)} pares={len(tr)} pos={ytr.mean():.3f} pos_weight={pos_weight:.2f}")
    log(f"val:   eventos={len(va.events)} pares={len(va)}")
    log(f"GPUs (DDP)={world}  lote efectivo={args.batch_size * args.grad_accum * world}")

    v = VLMVerifier(args.model, args.dtype, load_4bit=args.load_4bit, attn_impl=args.attn_impl,
                    max_pixels=args.max_pixels, device_map=device_map(world, local))
    coll = VLMCollator(v.processor, {"enable_thinking": False} if args.no_thinking else None)

    def evaluate_val(tag: str):
        """Cada proceso puntúa su porción de val; el informe lo calcula solo rank 0 (None en el resto)."""
        rows = gather_rows(score_dataset(v, va, coll, args.batch_size * 2, limit=args.val_limit,
                                         rank=rank, world=world), world)
        return report(va.events, rows, out, tag=tag) if main_proc else None

    # ---------------- prueba congelada (época 0) ----------------
    best_net = -math.inf
    last_dir = out / "last"
    resuming = args.resume and (last_dir / "state.pt").exists()
    if resuming and args.full_llm:
        raise SystemExit("--resume no está soportado con --full_llm")
    if not args.skip_frozen_eval and not resuming:
        log("== Evaluación congelada (zero-shot) ==")
        res = evaluate_val("_frozen")
        if main_proc:
            best_net = res["net"]
            print(f"neto congelado = {best_net}  (el fine-tuning debe superarlo para guardarse)")
        barrier(world)

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
    # lotes POR PROCESO: cada rank ve 1/world de los pares de cada época
    n_batches = len(tr) // (args.batch_size * world)
    total = args.epochs * n_batches // args.grad_accum
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * total), total)
    eval_every = max(1, n_batches // args.evals_per_epoch)

    step, start_ep, start_it = 0, 0, 0
    if resuming:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(v.model, load_file(str(last_dir / "adapter_model.safetensors")))
        st = torch.load(last_dir / "state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, start_ep, start_it, best_net = st["step"], st["ep"], st["it"], st["best_net"]
        log(f"== Reanudando: ep {start_ep}, lote {start_it}, step {step}, mejor neto {best_net}")

    # DDP envuelve el modelo (PEFT) para el paso de entrenamiento; la evaluación usa el modelo sin
    # envolver (sin colectivas en forward: cada rank puede tener un nº distinto de lotes de val).
    # Se crea DESPUÉS de cargar el adaptador al reanudar, así los pesos de los ranks coinciden.
    torch.manual_seed(args.seed + 1000 * rank)  # distintas máscaras de dropout por rank
    ddp = None
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp = DDP(v.model, device_ids=[local], output_device=local, broadcast_buffers=False,
                  find_unused_parameters=False)

    def save_last(ep, it_done):
        # gradiente acumulado a medias se descarta al reanudar (como mucho grad_accum-1 lotes)
        if it_done >= n_batches:
            ep, it_done = ep + 1, 0
        v.model.save_pretrained(last_dir)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "ep": ep, "it": it_done, "best_net": best_net}, last_dir / "state.pt")

    for ep in range(start_ep, args.epochs):
        v.model.train()
        run = []
        skip = start_it if ep == start_ep else 0
        # orden determinista por época (igual en todos los ranks): permite saltar lotes al reanudar
        perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(args.seed + ep))
        perm = perm[: n_batches * args.batch_size * world][rank::world]
        perm = perm[skip * args.batch_size:].tolist()
        dl = DataLoader(tr, batch_size=args.batch_size, sampler=perm, collate_fn=coll, num_workers=4)
        for it, batch in enumerate(dl, start=skip):
            do_step = (it + 1) % args.grad_accum == 0
            # no_sync: solo se sincroniza gradientes en el último micro-lote de cada acumulación
            with (ddp.no_sync() if ddp is not None and not do_step else contextlib.nullcontext()):
                loss, _ = v.loss(batch, pos_weight, model=ddp)
                (loss / args.grad_accum).backward()
            run.append(loss.item())
            if do_step:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    log(f"ep {ep} step {step}/{total} loss {np.mean(run[-50:]):.4f}")
            if (it + 1) % eval_every == 0 or it + 1 == n_batches:
                res = evaluate_val(f"_ep{ep}_it{it+1}")
                if main_proc:
                    if res["net"] > best_net:
                        best_net = res["net"]
                        v.model.save_pretrained(out / "best")
                        (out / "best" / "selection.json").write_text(json.dumps(res, indent=1))
                        print(f"** guardado: neto={best_net} (beta={res['beta']}, tau={res['tau']})")
                    save_last(ep, it + 1)
                barrier(world)
                v.model.train()
    log(f"mejor neto en validación: {best_net}")
    barrier(world)
    cleanup(world)


if __name__ == "__main__":
    main()

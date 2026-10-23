"""Paso 2: Dataset de pares (memoria congelada, candidato) + collator para el VLM.

Cada evento con M memorias y C candidatos produce M*C pares.
Etiqueta = 1 si el candidato es la misma identidad GT que la mayoría de la memoria.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from common import FrameCache, frame_path, read_jsonl, read_seqinfo, render_view

QUESTION = ("Is the person inside the red box in the NEW frame the same individual as the person "
            "inside the red box in the REFERENCE frames? Answer only Yes or No.")


def load_events(paths: list[str | Path]) -> list[dict]:
    evs = []
    for p in paths:
        evs += read_jsonl(Path(p))
    return evs


def expand_pairs(events: list[dict], min_purity: float = 0.0, drop_timeout: bool = False,
                 error_oversample: int = 1) -> list[tuple[int, int, int]]:
    """Lista de (idx_evento, idx_memoria, idx_candidato).

    error_oversample > 1 repite los eventos donde el tracker falló. Úsalo con cuidado:
    la distribución natural (mayoría de cruces bien resueltos) es la que enseña a NO romper.
    """
    pairs = []
    for ei, ev in enumerate(events):
        if drop_timeout and ev.get("timeout"):
            continue
        rep = error_oversample if ev["tracker_error"] else 1
        for mi, m in enumerate(ev["memories"]):
            if m["purity"] < min_purity:
                continue
            for ci in range(len(ev["candidates"])):
                pairs.extend([(ei, mi, ci)] * rep)
    return pairs


def pair_label(ev: dict, mi: int, ci: int) -> int:
    m, c = ev["memories"][mi], ev["candidates"][ci]
    return int(m["gt_major"] >= 0 and c["gt_id"] == m["gt_major"])


class CrossingPairDataset(Dataset):
    def __init__(self, event_files, view_mode="frame", n_mem=4, mem_stride=3, max_side=640,
                 crop_size=336, ctx=2.5, add_traj_text=False, min_purity=0.0, drop_timeout=False,
                 error_oversample=1):
        self.events = load_events(event_files)
        self.pairs = expand_pairs(self.events, min_purity, drop_timeout, error_oversample)
        self.view_mode, self.n_mem, self.mem_stride = view_mode, n_mem, mem_stride
        self.max_side, self.crop_size, self.ctx = max_side, crop_size, ctx
        self.add_traj_text = add_traj_text
        self._infos: dict[str, dict] = {}
        self._cache = FrameCache()

    def __len__(self):
        return len(self.pairs)

    def _info(self, seq_dir: str) -> dict:
        if seq_dir not in self._infos:
            self._infos[seq_dir] = read_seqinfo(Path(seq_dir))
        return self._infos[seq_dir]

    def _views(self, seq_dir: str, f: int, box) -> list:
        img = self._cache.get(frame_path(Path(seq_dir), self._info(seq_dir), f))
        return render_view(img, box, self.view_mode, self.max_side, self.ctx, self.crop_size)

    def _mem_indices(self, n: int) -> list[int]:
        idx = list(range(n - 1, -1, -self.mem_stride))[: self.n_mem]
        return idx[::-1]

    def __getitem__(self, i):
        ei, mi, ci = self.pairs[i]
        ev = self.events[ei]
        m, c = ev["memories"][mi], ev["candidates"][ci]
        info = self._info(ev["seq_dir"])
        W, H = info["width"], info["height"]

        content, images = [{"type": "text", "text": "REFERENCE frames (oldest to newest):"}], []
        for k in self._mem_indices(len(m["frames"])):
            for v in self._views(ev["seq_dir"], m["frames"][k], m["boxes"][k]):
                content.append({"type": "image"})
                images.append(v)
        content.append({"type": "text", "text": "NEW frame:"})
        for v in self._views(ev["seq_dir"], ev["query_frame"], c["box"]):
            content.append({"type": "image"})
            images.append(v)

        q = QUESTION
        if self.add_traj_text:
            def nb(b):
                return f"[{b[0]/W:.3f},{b[1]/H:.3f},{b[2]/W:.3f},{b[3]/H:.3f}]"
            traj = "; ".join(f"t={fr}: {nb(b)}" for fr, b in zip(m["frames"][-8:], m["boxes"][-8:]))
            q = (f"Reference trajectory (normalized boxes): {traj}. "
                 f"New box at t={ev['query_frame']}: {nb(c['box'])}. " + QUESTION)
        content.append({"type": "text", "text": q})

        return {
            "messages": [{"role": "user", "content": content}],
            "images": images,
            "label": pair_label(ev, mi, ci),
            "meta": {"event": ei, "mem": mi, "cand": ci,
                     "tracker_same": int(c["tid"] == m["tid"]),
                     "tracker_error": int(ev["tracker_error"]), "purity": m["purity"]},
        }


class VLMCollator:
    """Convierte una lista de items en tensores para el modelo (padding a la izquierda)."""

    def __init__(self, processor, chat_template_kwargs: dict | None = None):
        self.proc = processor
        self.proc.tokenizer.padding_side = "left"
        self.kw = chat_template_kwargs or {}

    def __call__(self, batch):
        import torch

        texts = [self.proc.apply_chat_template(b["messages"], tokenize=False,
                                               add_generation_prompt=True, **self.kw) for b in batch]
        images = [img for b in batch for img in b["images"]]
        inputs = self.proc(text=texts, images=images, padding=True, return_tensors="pt")
        inputs["labels_bin"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        inputs["meta"] = [b["meta"] for b in batch]
        return inputs


def split_by_sequence(event_files, val_frac=0.15, seed=0):
    """Si solo tienes un JSONL, separa por SECUENCIA (nunca por par) para no filtrar identidades."""
    evs = load_events(event_files)
    seqs = sorted({e["seq"] for e in evs})
    random.Random(seed).shuffle(seqs)
    val = set(seqs[: max(1, int(len(seqs) * val_frac))])
    return [e for e in evs if e["seq"] not in val], [e for e in evs if e["seq"] in val]

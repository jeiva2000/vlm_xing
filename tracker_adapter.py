"""Adaptador fino sobre BoxMOT.

La API de BoxMOT ha cambiado mucho entre versiones. Este es el ÚNICO archivo
que deberías tocar si tu versión instalada difiere. Contrato que usa el resto:

    trk = make_tracker("bytetrack" | "botsort" | "ocsort", **kwargs)
    ids, boxes = step(trk, dets_xyxy_score, frame_bgr)
        dets_xyxy_score: Nx5 [x1,y1,x2,y2,score]
        ids: (M,) int     boxes: Mx4 xyxy
"""
from __future__ import annotations

import numpy as np


def make_tracker(name: str, **kwargs):
    import boxmot

    # BoT-SORT usa ReID por defecto y exige pesos; aquí se desactiva (solo movimiento + CMC), así
    # no hay que descargar modelos y el resultado no depende de un extractor de apariencia.
    if name == "botsort":
        kwargs.setdefault("with_reid", False)

    # API nueva (boxmot >= ~v16): create_tracker("bytetrack", **overrides)
    try:
        return boxmot.create_tracker(name, **kwargs)
    except TypeError:
        pass
    # API antigua (v10-v13): create_tracker(tracker_type, tracker_config, reid_weights, device, half)
    from pathlib import Path

    cfg = kwargs.pop("tracker_config", None)
    if cfg is None:
        from boxmot.utils import TRACKER_CONFIGS  # type: ignore
        cfg = Path(TRACKER_CONFIGS) / f"{name}.yaml"
    extra = {k: kwargs.pop(k) for k in ("with_reid",) if k in kwargs}
    if extra:  # la API antigua solo lee parámetros del yaml: se escribe una copia con los overrides
        import tempfile

        import yaml

        with open(cfg) as fh:
            conf = yaml.safe_load(fh)
        for k, v in extra.items():
            conf[k] = {"type": "choice", "default": v, "options": [v]}
        tmp = tempfile.NamedTemporaryFile("w", suffix=f"_{name}.yaml", delete=False)
        yaml.safe_dump(conf, tmp)
        tmp.close()
        cfg = tmp.name
    return boxmot.create_tracker(
        name, cfg,
        kwargs.pop("reid_weights", None),
        kwargs.pop("device", "cpu"),
        kwargs.pop("half", False),
    )


def step(trk, dets: np.ndarray, frame_bgr: np.ndarray):
    if len(dets) == 0:
        dets6 = np.zeros((0, 6), dtype=np.float32)
    else:
        cls = np.zeros((len(dets), 1), dtype=np.float32)  # clase única: persona
        dets6 = np.hstack([dets[:, :5], cls]).astype(np.float32)
    out = trk.update(dets6, frame_bgr)
    out = np.asarray(out)
    if out.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    # Columnas clásicas de salida: [x1,y1,x2,y2,id,conf,cls,det_ind]
    return out[:, 4].astype(np.int64), out[:, :4].astype(np.float32)

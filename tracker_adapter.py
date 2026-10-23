"""Adaptador fino sobre BoxMOT.

La API de BoxMOT ha cambiado mucho entre versiones. Este es el ÚNICO archivo
que deberías tocar si tu versión instalada difiere. Contrato que usa el resto:

    trk = make_tracker("bytetrack" | "botsort" | "ocsort", **kwargs)
    ids, boxes = step(trk, dets_xyxy_score, frame_bgr)
        dets_xyxy_score: Nx5 [x1,y1,x2,y2,score]
        ids: (M,) int     boxes: Mx4 xyxy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_REID = Path("models") / "osnet_x0_25_msmt17.pt"
# mismo enlace que usa BoxMOT (TRAINED_URLS); se prefiere el de la versión instalada si existe
_REID_URLS = {"osnet_x0_25_msmt17.pt": "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"}


def ensure_reid_weights(path: str | Path = DEFAULT_REID) -> Path:
    """Devuelve la ruta a los pesos ReID, descargándolos si no existen (necesita internet)."""
    path = Path(path)
    if path.is_file():
        return path
    url = None
    for mod in ("boxmot.reid.core.config", "boxmot.appearance.reid.config"):  # según versión
        try:
            url = __import__(mod, fromlist=["TRAINED_URLS"]).TRAINED_URLS.get(path.name)
        except Exception:
            continue
        if url:
            break
    url = url or _REID_URLS.get(path.name)
    if not url:
        raise FileNotFoundError(f"No existen los pesos ReID {path} y no se conoce su URL; descárgalos a mano")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando pesos ReID -> {path}")
    import gdown  # dependencia de boxmot

    got = gdown.download(url, str(path), quiet=False)
    if not got or not path.is_file():
        raise RuntimeError(f"Falló la descarga de {url}; descarga el archivo a mano en {path}")
    return path


def make_tracker(name: str, reid: bool = False, reid_weights=None, **kwargs):
    """reid / reid_weights solo afectan a BoT-SORT (con reid=False va solo con movimiento + CMC)."""
    import boxmot

    if name == "botsort":
        kwargs["with_reid"] = bool(reid)
        if reid:
            import torch

            cuda = torch.cuda.is_available()
            kwargs["reid_weights"] = ensure_reid_weights(reid_weights or DEFAULT_REID)
            kwargs.setdefault("device", "cuda:0" if cuda else "cpu")
            kwargs.setdefault("half", cuda)

    # API nueva (boxmot >= ~v16): create_tracker("bytetrack", **overrides)
    try:
        return boxmot.create_tracker(name, **kwargs)
    except TypeError:
        pass
    # API antigua (v10-v13): create_tracker(tracker_type, tracker_config, reid_weights, device, half)
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

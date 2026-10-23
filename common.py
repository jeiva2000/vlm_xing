"""Utilidades comunes: lectura DanceTrack, cajas, render de imágenes y registro de modelos."""
from __future__ import annotations

import configparser
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Registro de variantes de modelo. Añade aquí las que quieras probar.
# ---------------------------------------------------------------------------
MODEL_VARIANTS = {
    "qwen3vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
    # Qwen3.5: verificar nombre exacto en HF y desactivar el modo "thinking".
    "qwen35-4b": "Qwen/Qwen3.5-4B",
    "qwen35-9b": "Qwen/Qwen3.5-9B",
}


def resolve_model(name: str) -> str:
    """Acepta un alias del registro o una ruta/ID de HF directamente."""
    return MODEL_VARIANTS.get(name, name)


# ---------------------------------------------------------------------------
# DanceTrack / formato MOT
# ---------------------------------------------------------------------------
def read_seqinfo(seq_dir: Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(seq_dir / "seqinfo.ini")
    s = cp["Sequence"]
    return {
        "name": s.get("name", seq_dir.name),
        "length": int(s["seqLength"]),
        "width": int(s["imWidth"]),
        "height": int(s["imHeight"]),
        "img_dir": s.get("imDir", "img1"),
        "img_ext": s.get("imExt", ".jpg"),
    }


def frame_path(seq_dir: Path, info: dict, f: int) -> Path:
    return seq_dir / info["img_dir"] / f"{f:08d}{info['img_ext']}"


def read_mot_txt(path: Path, is_gt: bool) -> dict[int, np.ndarray]:
    """Devuelve {frame: array Nx6}. GT: [id,x1,y1,x2,y2,1]. Dets: [-1,x1,y1,x2,y2,score]."""
    out = defaultdict(list)
    if not path.exists():
        return {}
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    for r in data:
        f, tid, x, y, w, h, s = int(r[0]), int(r[1]), *r[2:7]
        if is_gt and len(r) > 6 and r[6] == 0:  # entradas ignoradas
            continue
        out[f].append([tid, x, y, x + w, y + h, 1.0 if is_gt else s])
    return {f: np.asarray(v, dtype=np.float32) for f, v in out.items()}


def list_sequences(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / "seqinfo.ini").exists())


# ---------------------------------------------------------------------------
# Cajas
# ---------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre cajas xyxy, a: Nx4, b: Mx4."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_to_gt(boxes: np.ndarray, gt: np.ndarray | None, thr: float = 0.5) -> np.ndarray:
    """Asigna a cada caja del tracker un id GT (o -1) por Hungarian sobre IoU."""
    ids = np.full(len(boxes), -1, dtype=np.int64)
    if gt is None or len(gt) == 0 or len(boxes) == 0:
        return ids
    iou = iou_matrix(boxes, gt[:, 1:5])
    r, c = linear_sum_assignment(-iou)
    for i, j in zip(r, c):
        if iou[i, j] >= thr:
            ids[i] = int(gt[j, 0])
    return ids


# ---------------------------------------------------------------------------
# Render de imágenes para el VLM
# ---------------------------------------------------------------------------
BOX_COLOR = (255, 0, 0)  # rojo en RGB


def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def render_view(img_rgb: np.ndarray, box, mode: str = "frame", max_side: int = 640,
                ctx: float = 2.5, crop_size: int = 336, thickness: int = 4) -> list[Image.Image]:
    """Dibuja SOLO la caja indicada.

    mode:
      frame -> frame completo con la caja, redimensionado a max_side.
      crop  -> recorte con contexto (ctx veces el tamaño de la caja).
      both  -> [crop, frame]  (dos imágenes por vista).
    """
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    canvas = img_rgb.copy()
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, thickness)
    views = []
    if mode in ("crop", "both"):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = (x2 - x1) * ctx / 2, (y2 - y1) * ctx / 2
        H, W = canvas.shape[:2]
        a, b = int(max(0, cx - hw)), int(max(0, cy - hh))
        c, d = int(min(W, cx + hw)), int(min(H, cy + hh))
        crop = canvas[b:d, a:c]
        views.append(Image.fromarray(_resize_max(crop, crop_size)))
    if mode in ("frame", "both"):
        views.append(Image.fromarray(_resize_max(canvas, max_side)))
    return views


class FrameCache:
    """Caché LRU simple de frames RGB (varias vistas reutilizan el mismo frame)."""

    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self._d: dict[str, np.ndarray] = {}

    def get(self, path: Path) -> np.ndarray:
        k = str(path)
        if k in self._d:
            img = self._d.pop(k)
        else:
            bgr = cv2.imread(k)
            if bgr is None:
                raise FileNotFoundError(k)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._d[k] = img
        if len(self._d) > self.max_items:
            self._d.pop(next(iter(self._d)))
        return img


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

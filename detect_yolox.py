"""Detecciones con YOLOX-X entrenado en DanceTrack (descarga automática de pesos).

Pesos: los mismos que usa BoxMOT para su detector "yolox-x-dancetrack"
(mirror en Hugging Face). El original es `bytetrack_dance_model.pth.tar`
del repo de DanceTrack/ByteTrack (Google Drive); si lo tienes, pásalo con --weights.

Salida: un .txt por secuencia en formato MOT  frame,-1,x,y,w,h,score,-1,-1,-1

Uso independiente:
  python detect_yolox.py --data_root $DT/val --out_dir dets/yolox_dancetrack/val --fp16
(mine_events.py lo llama solo si faltan detecciones, con --det_source yolox)
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from common import frame_path, list_sequences, read_seqinfo

HF_REPO, HF_FILE = "Lekim89/yolox", "yolox_x_dancetrack.pt"
WEIGHTS_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILE}"
DEFAULT_WEIGHTS = Path("models") / HF_FILE
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def ensure_weights(path: str | Path = DEFAULT_WEIGHTS) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando pesos YOLOX DanceTrack -> {path}")
    try:
        from huggingface_hub import hf_hub_download
        got = Path(hf_hub_download(HF_REPO, HF_FILE, local_dir=str(path.parent)))
        if got != path:
            got.replace(path)
    except Exception as e:  # sin huggingface_hub o sin acceso: descarga directa
        print(f"  hf_hub_download falló ({e}); probando descarga directa")
        urllib.request.urlretrieve(WEIGHTS_URL, path)
    return path


class YoloXDanceTrack:
    def __init__(self, weights=DEFAULT_WEIGHTS, device="cuda", fp16=True,
                 img_size=(800, 1440), conf=0.01, nms=0.7):
        import torch
        from yolox.exp import get_exp
        from yolox.utils.model_utils import fuse_model

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device.type == "cuda"
        self.img_size, self.conf, self.nms = tuple(img_size), conf, nms
        exp = get_exp(None, "yolox_x")  # misma arquitectura que el modelo de ByteTrack/DanceTrack
        exp.num_classes = 1
        model = exp.get_model()
        ckpt = torch.load(str(ensure_weights(weights)), map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        model = fuse_model(model.eval()).to(self.device)
        self.model = model.half() if self.fp16 else model

    def _prep(self, bgr: np.ndarray):
        h, w = self.img_size
        r = min(h / bgr.shape[0], w / bgr.shape[1])
        rs = cv2.resize(bgr, (int(bgr.shape[1] * r), int(bgr.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
        pad = np.full((h, w, 3), 114.0, dtype=np.float32)
        pad[: rs.shape[0], : rs.shape[1]] = rs
        pad = (pad[:, :, ::-1] / 255.0 - MEAN) / STD  # preprocesado de ByteTrack (BGR->RGB, norm.)
        return np.ascontiguousarray(pad.transpose(2, 0, 1), dtype=np.float32), r

    def __call__(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Devuelve por frame un array Nx5 [x1,y1,x2,y2,score] en píxeles originales."""
        from yolox.utils import postprocess

        torch = self.torch
        ims, ratios = zip(*(self._prep(f) for f in frames_bgr))
        x = torch.from_numpy(np.stack(ims)).to(self.device)
        x = x.half() if self.fp16 else x
        with torch.no_grad():
            out = self.model(x)
        out = postprocess(out.float().clone(), 1, self.conf, self.nms)
        res = []
        for o, r in zip(out, ratios):
            if o is None:
                res.append(np.zeros((0, 5), np.float32))
                continue
            o = o.cpu().numpy()
            boxes = o[:, :4] / r
            score = o[:, 4] * o[:, 5]
            res.append(np.hstack([boxes, score[:, None]]).astype(np.float32))
        return res


def detect_sequence(det: YoloXDanceTrack, seq_dir: Path, out_txt: Path, batch: int = 4):
    info = read_seqinfo(seq_dir)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    frames = list(range(1, info["length"] + 1))
    for i in range(0, len(frames), batch):
        fs = frames[i: i + batch]
        imgs = [cv2.imread(str(frame_path(seq_dir, info, f))) for f in fs]
        for f, d in zip(fs, det(imgs)):
            for x1, y1, x2, y2, s in d:
                lines.append(f"{f},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.4f},-1,-1,-1")
    tmp = out_txt.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out_txt)  # escritura atómica: un corte no deja archivos a medias
    print(f"  {seq_dir.name}: {len(lines)} detecciones -> {out_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--img_size", type=int, nargs=2, default=[800, 1440])
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--nms", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    det = YoloXDanceTrack(args.weights, args.device, args.fp16, args.img_size, args.conf, args.nms)
    for s in list_sequences(Path(args.data_root)):
        out = Path(args.out_dir) / f"{s.name}.txt"
        if out.exists() and not args.overwrite:
            continue
        detect_sequence(det, s, out, args.batch)


if __name__ == "__main__":
    main()

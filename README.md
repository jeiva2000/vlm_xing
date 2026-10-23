# vlm_xing — VLM como verificador de identidad en cruces

Flujo:
0. `detect_yolox.py` -> YOLOX-X entrenado en DanceTrack; descarga los pesos solo y guarda
   detecciones en formato MOT. `mine_events.py` lo ejecuta automáticamente si faltan.
1. `mine_events.py`  -> ejecuta ByteTrack/BoT-SORT/OC-SORT (BoxMOT), detecta cruces SIN GT,
   congela memorias y guarda eventos (JSONL). Imprime el recall del disparador (P0).
2. `evaluate.py`     -> prueba congelada (zero-shot): puntúa pares y calcula arreglados/rotos/neto
   con barrido de beta (sesgo hacia el tracker) y tau.
3. `train_lora.py`   -> época 0 congelada + LoRA; guarda solo si mejora el NETO en validación.

Apoyo: `common.py` (IO, render, registro de modelos), `tracker_adapter.py` (único punto que
depende de la versión de BoxMOT), `dataset.py`, `verifier.py`, `assoc.py`.

## Instalación
Ver la cabecera de `requirements.txt`: primero torch (cu118 en el cluster), luego
`pip install -r requirements.txt`, y por último `pip install --no-build-isolation --no-deps yolox==0.3.0`.

## Ejemplo
```bash
# eventos de train y val: detecta con YOLOX DanceTrack (si faltan) y ejecuta los trackers
python mine_events.py --data_root $DT/train --det_dir dets/yolox_dancetrack/train --out_dir events/train --fp16
python mine_events.py --data_root $DT/val   --det_dir dets/yolox_dancetrack/val   --out_dir events/val --fp16

# zero-shot con distintas variantes / vistas
python evaluate.py --model qwen3vl-2b --events events/val/events_bytetrack.jsonl --view_mode crop --out runs/zs2b_crop
python evaluate.py --model qwen3vl-8b --events events/val/events_bytetrack.jsonl --view_mode both --add_traj_text --out runs/zs8b_both_traj

# fine-tuning
python train_lora.py --model qwen3vl-8b --train_events events/train/events_*.jsonl \
    --val_events events/val/events_bytetrack.jsonl --out runs/lora8b
```
Depuración en la RTX 2080: `--model qwen3vl-2b --dtype float16 --load_4bit`.
Sin detector: `--det_source gt --gt_jitter 3 --gt_drop 0.05` (solo para depurar, no para resultados).

## SLURM (1 GPU)
```bash
# 1) en el nodo login (con internet): crea el entorno y descarga YOLOX + el VLM
DT_ROOT=/ruta/DanceTrack bash run_vlm_xing.slurm
# 2) pipeline completo: minar -> zero-shot -> LoRA -> evaluación final
DT_ROOT=/ruta/DanceTrack sbatch run_vlm_xing.slurm
# variantes
DT_ROOT=... MODEL=qwen3vl-4b VIEW_MODE=both EXTRA_DATA_ARGS=--add_traj_text sbatch run_vlm_xing.slurm
DT_ROOT=... STAGES="zeroshot" MODEL=qwen3vl-2b sbatch run_vlm_xing.slurm
```
El val se divide por secuencia en `*.sel.jsonl` (elegir checkpoint y beta) y `*.test.jsonl` (reportar).

## Varias GPUs (DDP)
`train_lora.py` y `evaluate.py` usan DDP bajo `torchrun` (un modelo completo por GPU, así que el
modelo debe caber en una GPU; con 8B en bf16 caben en una A40 con LoRA + gradient checkpointing):
```bash
torchrun --standalone --nproc_per_node 4 train_lora.py --model qwen3vl-8b ... --grad_accum 2
```
`--grad_accum` es POR GPU: lote efectivo = batch_size × grad_accum × nº GPUs. `run_vlm_xing.slurm`
detecta las GPUs asignadas y ajusta `GRAD_ACCUM` para mantener un lote efectivo de 16.
La validación se reparte entre procesos; rank 0 calcula métricas y guarda `best/` y `last/`.
Sin `torchrun` todo funciona como antes (1 proceso, `device_map=auto`).

## BoT-SORT con ReID
Por defecto BoT-SORT va sin ReID (solo movimiento + CMC). Con `--reid` (o `REID=1` en el `.slurm`/smoke test)
usa apariencia con `models/osnet_x0_25_msmt17.pt`, que se descarga si falta (gdown, necesita internet; se puede
cambiar con `--reid_weights`). Los eventos de BoT-SORT con y sin ReID no son comparables: usa `--out_dir` distintos.

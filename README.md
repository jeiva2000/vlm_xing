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

## TrackEval (tracker solo vs tracker + VLM)
`apply_vlm.py` aplica las decisiones del VLM (assoc.decide con el beta/tau de `best/selection.json`) sobre las
trayectorias crudas del tracker: desde el frame de la consulta, el candidato elegido hereda el ID de la memoria
(post-proceso, los frames del cruce no se reescriben). `eval_tracks.py` evalúa con TrackEval (HOTA, DetA, AssA,
MOTA, IDF1, IDSW) las secuencias de val.test para el tracker solo y para tracker+VLM:
```bash
STAGES="trackeval" sbatch --export=NONE,DT_ROOT=... run_vlm_xing.slurm   # tras mine/train/eval
python eval_tracks.py --setup   # clona TrackEval en third_party/ (git + internet) y lo parchea para numpy>=1.24
```
`mine_events.py` guarda ahora `tracks_<tracker>/<seq>.txt` (salida cruda); si no existen se regeneran solo para val.test.
Resultados: `work/runs/trackeval_<tag>/<tracker>/metrics_trackeval.json`. Solo se evalúa el adaptador entrenado
(beta/tau elegidos en val.sel); las secuencias sin eventos no entran en la evaluación.

## Imágenes de casos (acierta / falla)
`visualize_cases.py` clasifica cada memoria de val.test en `fixed` (el VLM arregla al tracker), `broken` (lo rompe),
`correct` (ambos aciertan) y `wrong` (ambos fallan), y guarda una imagen por caso: frames de referencia + candidatos
del frame de consulta con su score (borde verde = candidato correcto; etiquetas TRUTH / TRK / VLM). Abre
`work/runs/cases_<tag>/<tracker>/index.html`. Etapa `viz` del `.slurm` (`MAX_CASES` = imágenes por categoría):
```bash
STAGES="viz" sbatch --export=NONE,DT_ROOT=... run_vlm_xing.slurm
```
Para el zero-shot: `python visualize_cases.py --events ... --scores runs/zs_*/scores.jsonl --beta 2 --tau 0 --out ...`.

## Entrenamiento amplio (más épocas, un job por modelo, checkpoints)
```bash
DT_ROOT=/ruta/DanceTrack bash launch_wide.sh      # login; lanza un job por modelo, encadenados
DT_ROOT=... MODELS="qwen3vl-4b qwen3vl-8b" EPOCHS=10 PATIENCE=8 CHAIN=0 bash launch_wide.sh
```
Usa `PROFILE=wide` del `.slurm` (8 épocas, `--save_all`, parada temprana con paciencia 6 evaluaciones, sufijo `_e<EPOCHS>` en el
nombre del run para no mezclarse con los cortos). Cada run en `work/runs/lora_<modelo>_<vista>_m<N>_e<EPOCHS>/` contiene:
`best/` (mejor neto en val.sel), `last/` (para `--resume`), `ckpt_ep<E>_it<I>/` (adaptador de CADA evaluación) y
`history.jsonl` (pérdida train/val, AUC y neto por evaluación). Curvas: `python plot_history.py work/runs/lora_*_e8`.
Un checkpoint concreto se evalúa con `evaluate.py --adapter .../ckpt_ep3_it500 ...`. `--resume` se niega a continuar si la
configuración (épocas, lr, lote, nº de GPUs...) cambió respecto a `last/`: usa otro `--out`. Tras un corte por tiempo,
relanza el mismo comando y continúa. Con el modelo entero por GPU (DDP) caben 2B/4B/8B; 32B no.

## Matriz consulta vs banco

`visualize_matrix.py` dibuja una imagen por **evento** con la matriz completa de asociaciones: filas = banco
(memorias, con sus frames de referencia), columnas = consulta (candidatos del frame de la consulta), celdas =
score del VLM (rojo → verde). Borde verde = verdad, `T` = elección del tracker, `V` = elección del VLM.
Se agrupan por su peor caso (`broken > wrong > fixed > correct`). La etapa `viz` la genera en
`work/runs/matrix_<tag>/<tracker>/index.html`. A mano:

```bash
python visualize_matrix.py --events work/events/val/events_botsort.test.jsonl \
  --scores work/runs/final_<tag>_botsort/scores.jsonl --selection work/runs/sel_<tag>/selection.json \
  --out matrix_botsort --cats broken wrong           # --seq dancetrack0063 --event 120 para uno concreto
```

## Más datos y ejemplos difíciles

La pérdida de validación sube mientras la de train baja (sobreajuste), así que más épocas no ayudan: hace falta
más diversidad. Tres opciones, combinables, y cada una añade su marca al nombre del run (`_h3`, `_aug`, `_v3`):

- `HARD_OVERSAMPLE=3` repite 3 veces los pares clave de las memorias donde el tracker erró: el candidato que
  eligió mal (negativo difícil) y el correcto (positivo difícil). Es más fino que `--error_oversample`, que repite eventos enteros.
- `AUGMENT=1` aumentación solo en train: giro horizontal (el mismo en todas las vistas del par), brillo/contraste,
  contexto del recorte y separación temporal de la memoria. Con `--add_traj_text` el giro se desactiva.
- `MINE_VARIANTS="ms03+ms005+ti01"` vuelve a minar **solo train** con otros umbrales, reusando las detecciones de YOLOX
  (solo cuesta el tracking): `ms03` y `ms005` cambian el umbral de detección (0.3 / 0.05) y `ti01` baja el umbral de
  disparo (más eventos). Salen como `events_<tracker>_<variante>.jsonl` y entran solos en el entrenamiento.

```bash
DT_ROOT=... MODELS="qwen3vl-4b" EPOCHS=3 HARD_OVERSAMPLE=3 AUGMENT=1 MINE_VARIANTS="ms03+ms005" bash launch_wide.sh
```

Límite: las variantes siguen saliendo de las mismas 40 secuencias de DanceTrack; añaden otros errores del tracker,
no identidades nuevas. Para eso haría falta otro dataset. val.sel/val.test no se tocan, así que los resultados siguen siendo comparables.

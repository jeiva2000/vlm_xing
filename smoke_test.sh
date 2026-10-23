#!/bin/bash
# Prueba de humo: 1 secuencia, detecciones sintéticas desde GT (solo depuración), 2B en 4 bits.
# Uso: DT_ROOT=/ruta/DanceTrack [SEQ=dancetrack0001] bash smoke_test.sh
# Comprueba mine -> evaluate -> train (1 época corta) -> reanudación, sin resultados válidos.
set -euo pipefail
DT_ROOT="${DT_ROOT:?define DT_ROOT}"
SEQ="${SEQ:-$(ls "$DT_ROOT/train" | head -1)}"
OUT="${OUT:-smoke_out}"
M=(--model qwen3vl-2b --dtype float16 --load_4bit --n_mem 2)
VAL="$OUT/ev/val"; TR="$OUT/ev/train"

mkdir -p "$OUT"
# mismas secuencia en train/val no sirve (el assert lo impide): usa dos secuencias distintas
SEQ2="${SEQ2:-$(ls "$DT_ROOT/val" | head -1)}"
python mine_events.py --data_root "$DT_ROOT/train" --seqs "$SEQ"  --det_source gt --out_dir "$TR"  --trackers bytetrack botsort ocsort
python mine_events.py --data_root "$DT_ROOT/val"   --seqs "$SEQ2" --det_source gt --out_dir "$VAL" --trackers bytetrack
python evaluate.py "${M[@]}" --events "$VAL/events_bytetrack.jsonl" --limit 40 --out "$OUT/zs"
python train_lora.py "${M[@]}" --train_events "$TR"/events_*.jsonl --val_events "$VAL/events_bytetrack.jsonl" \
  --val_limit 40 --epochs 1 --batch_size 1 --grad_accum 2 --out "$OUT/lora" --skip_frozen_eval
python train_lora.py "${M[@]}" --train_events "$TR"/events_*.jsonl --val_events "$VAL/events_bytetrack.jsonl" \
  --val_limit 40 --epochs 2 --batch_size 1 --grad_accum 2 --out "$OUT/lora" --skip_frozen_eval --resume
echo "SMOKE OK"

#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=e2e_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:4             # TODAS las GPUs del trabajo en este nodo (p. ej. gpu:a100:5); las dos etapas usan DDP con todas. Comprueba los nombres con: sinfo -o "%N %G"
##SBATCH --nodelist=gpu02            # opcional: fija el nodo (comprueba antes con: scontrol show node gpu02)
#SBATCH --cpus-per-task=17           # unos 8 por GPU
#SBATCH --mem=160G                   # unos 32G por GPU
#SBATCH -t 72:00:00
# Prueba de humo: 1 secuencia, detecciones sintéticas desde GT (solo depuración), 2B en 4 bits.
# Local:   DT_ROOT=/ruta/DanceTrack [SEQ=dancetrack0001] bash smoke_test.sh
# SLURM:   mkdir -p logs && sbatch --export=NONE,DT_ROOT=/ruta/DanceTrack smoke_test.sh
#          (con --export=NONE el entorno no se hereda: pasa DT_ROOT/SEQ/ENV_DIR/MODULES así)
# Comprueba mine -> evaluate -> train (1 época corta) -> reanudación, sin resultados válidos.
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then  # --export=NONE: cargar módulos y entorno a mano
  cd "${SLURM_SUBMIT_DIR:-.}"
  if command -v module >/dev/null 2>&1; then
    module purge || true
    # shellcheck disable=SC2086
    module load ${MODULES:-GCC/12.2.0 CUDA/11.8.0} ${PYTHON_MODULE:-}
  fi
  # shellcheck disable=SC1091
  source "${ENV_DIR:-$HOME/envs/vlm_xing}/bin/activate"
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
fi
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

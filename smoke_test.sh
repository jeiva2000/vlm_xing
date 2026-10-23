#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=e2e_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:2             # 2 GPUs para ejercitar DDP (con 1 también funciona). Comprueba los nombres con: sinfo -o "%N %G"
##SBATCH --nodelist=gpu02            # opcional: fija el nodo (comprueba antes con: scontrol show node gpu02)
#SBATCH --cpus-per-task=16           # unos 8 por GPU
#SBATCH --mem=64G                    # unos 32G por GPU
#SBATCH -t 02:00:00
# Prueba de humo: 1 secuencia, detecciones sintéticas desde GT (solo depuración), 2B en 4 bits.
# Local:   DT_ROOT=/ruta/DanceTrack [SEQ=dancetrack0001] bash smoke_test.sh
# SLURM:   mkdir -p logs && sbatch --export=NONE,DT_ROOT=/ruta/DanceTrack smoke_test.sh
#          (con --export=NONE el entorno no se hereda: pasa DT_ROOT/SEQ/CONDA_ENV/MODULES así)
# Comprueba mine -> evaluate -> train (1 época corta) -> reanudación, sin resultados válidos.
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then  # --export=NONE: cargar módulos y entorno a mano
  cd "${SLURM_SUBMIT_DIR:-.}"
  # shellcheck disable=SC1091
  source "$SLURM_SUBMIT_DIR/cluster_env.sh"    # módulos GCC/CUDA + variables
  # shellcheck disable=SC1091
  source "$SLURM_SUBMIT_DIR/setup_env.sh"      # crea el entorno si no existe (CONDA_ENV, def. vlm_xing); si existe, lo activa
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
fi
DT_ROOT="${DT_ROOT:?define DT_ROOT}"
SEQ="${SEQ:-$(ls "$DT_ROOT/train" | head -1)}"
OUT="${OUT:-smoke_out}"
NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
if (( NGPU > 1 )); then LAUNCH=(torchrun --standalone --nproc_per_node "$NGPU"); else LAUNCH=(python); fi
echo "smoke: GPUs=$NGPU"
M=(--model qwen3vl-2b --dtype float16 --load_4bit --n_mem 2)
VAL="$OUT/ev/val"; TR="$OUT/ev/train"

mkdir -p "$OUT" logs
# mismas secuencia en train/val no sirve (el assert lo impide): usa dos secuencias distintas
SEQ2="${SEQ2:-$(ls "$DT_ROOT/val" | head -1)}"
python mine_events.py --data_root "$DT_ROOT/train" --seqs "$SEQ"  --det_source gt --out_dir "$TR"  --trackers bytetrack botsort ocsort $([[ "${REID:-0}" == 1 ]] && echo --reid)
python mine_events.py --data_root "$DT_ROOT/val"   --seqs "$SEQ2" --det_source gt --out_dir "$VAL" --trackers bytetrack
"${LAUNCH[@]}" evaluate.py "${M[@]}" --events "$VAL/events_bytetrack.jsonl" --limit 40 --out "$OUT/zs"
"${LAUNCH[@]}" train_lora.py "${M[@]}" --train_events "$TR"/events_*.jsonl --val_events "$VAL/events_bytetrack.jsonl" \
  --val_limit 40 --epochs 1 --batch_size 1 --grad_accum 2 --out "$OUT/lora" --skip_frozen_eval
"${LAUNCH[@]}" train_lora.py "${M[@]}" --train_events "$TR"/events_*.jsonl --val_events "$VAL/events_bytetrack.jsonl" \
  --val_limit 40 --epochs 2 --batch_size 1 --grad_accum 2 --out "$OUT/lora" --skip_frozen_eval --resume
echo "SMOKE OK"

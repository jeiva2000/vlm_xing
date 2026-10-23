#!/bin/bash
#SBATCH --job-name=vlm_xing
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:4                # nº de GPUs: zero-shot, LoRA y eval usan DDP con todas. Comprueba: sinfo -o "%N %G"
#SBATCH --cpus-per-task=32              # unos 8 por GPU
#SBATCH --mem=160G                      # unos 32-40G por GPU
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ---------------------------------------------------------------------------
# Uso:
#   bash  run_vlm_xing.slurm      -> en el nodo login: crea/actualiza el entorno y descarga
#                                    pesos (YOLOX + VLM). Útil si los nodos GPU no tienen internet.
#   sbatch run_vlm_xing.slurm     -> pipeline completo en 1 GPU.
#   STAGES="zeroshot train" MODEL=qwen3vl-4b sbatch run_vlm_xing.slurm   -> solo algunas etapas.
# ---------------------------------------------------------------------------
set -euo pipefail

# ============================ CONFIGURACIÓN ================================
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}"
WORK_DIR="${WORK_DIR:-$CODE_DIR/work}"           # detecciones, eventos, runs
DT_ROOT="${DT_ROOT:-/path/to/DanceTrack}"         # AJUSTAR: contiene train/ y val/
ENV_DIR="${ENV_DIR:-$HOME/envs/vlm_xing}"        # solo si no hay conda (venv)
CONDA_ENV="${CONDA_ENV:-vlm_xing}"                # entorno de miniconda (activate_env.sh)
PYTHON_BIN="${PYTHON_BIN:-python3}"               # debe ser Python >= 3.10
PYTHON_MODULE="${PYTHON_MODULE:-}"                # p.ej. "Python/3.10.8-GCCcore-12.2.0" si hace falta
MODULES="${MODULES:-GCC/12.2.0 CUDA/11.8.0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"   # en almacenamiento compartido

MODEL="${MODEL:-qwen3vl-8b}"
TRACKERS="${TRACKERS:-bytetrack botsort ocsort}"
VAL_TRACKER="${VAL_TRACKER:-bytetrack}"           # tracker cuyos eventos val.sel eligen el checkpoint
VIEW_MODE="${VIEW_MODE:-frame}"
N_MEM="${N_MEM:-4}"
REID="${REID:-0}"                                 # 1 = BoT-SORT con ReID (descarga osnet si falta)
EXTRA_DATA_ARGS="${EXTRA_DATA_ARGS:-}"            # p.ej. "--add_traj_text"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-}"                     # POR GPU; vacío = auto (lote efectivo 16 repartido entre las GPUs)
STAGES="${STAGES:-mine zeroshot train eval}"

TORCH_SPEC="torch==2.7.1 torchvision==0.22.1"     # última serie con wheels cu118
TORCH_INDEX="https://download.pytorch.org/whl/cu118"
# ===========================================================================

log() { echo "[$(date '+%F %T')] $*"; }
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

mkdir -p "$CODE_DIR/logs" "$WORK_DIR"
cd "$CODE_DIR"

# ------------------------------- módulos -----------------------------------
# shellcheck disable=SC1091
source "$CODE_DIR/cluster_env.sh"

# ------------------------------- entorno -----------------------------------
# Crea el entorno (conda) si no existe e instala dependencias; si existe, solo lo activa. Ver setup_env.sh.
# shellcheck disable=SC1091
source "$CODE_DIR/setup_env.sh"

# shellcheck disable=SC1091
source "$CODE_DIR/activate_env.sh"

python - <<EOF
import torch, transformers
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU disponible:", torch.cuda.is_available())
print("transformers", transformers.__version__)
EOF

# --------------------------- descarga de pesos ------------------------------
log "Comprobando pesos (YOLOX DanceTrack y $MODEL)"
python - <<EOF
from detect_yolox import ensure_weights
from common import resolve_model
from huggingface_hub import snapshot_download
ensure_weights("models/yolox_x_dancetrack.pt")
snapshot_download(resolve_model("$MODEL"))
EOF

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  log "Fuera de SLURM: entorno y pesos listos. Lanza el pipeline con: sbatch $0"
  exit 0
fi

for split in train val; do
  [[ -f "$DT_ROOT/$split/$(ls "$DT_ROOT/$split" 2>/dev/null | head -1)/seqinfo.ini" ]] \
    || { echo "DT_ROOT inválido ($DT_ROOT): se esperan $DT_ROOT/{train,val}/<seq>/seqinfo.ini"; exit 1; }
done

export HF_HUB_OFFLINE=1   # en el nodo GPU todo debe estar ya descargado
nvidia-smi --query-gpu=name,memory.total --format=csv
NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
if (( NGPU > 1 )); then
  LAUNCH=(torchrun --standalone --nproc_per_node "$NGPU")   # DDP: un proceso (y un modelo completo) por GPU
else
  LAUNCH=(python)
fi
[[ -n "$GRAD_ACCUM" ]] || GRAD_ACCUM=$(( 16 / (BATCH * NGPU) > 0 ? 16 / (BATCH * NGPU) : 1 ))
DTYPE=bfloat16
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_bf16_supported() else 1)' || DTYPE=float16
log "dtype=$DTYPE  modelo=$MODEL  etapas=[$STAGES]  GPUs=$NGPU  grad_accum/GPU=$GRAD_ACCUM"

EV="$WORK_DIR/events"
DETS="$WORK_DIR/dets/yolox_dancetrack"
RUN_TAG="${MODEL}_${VIEW_MODE}_m${N_MEM}"
# shellcheck disable=SC2206
DATA_ARGS=(--view_mode "$VIEW_MODE" --n_mem "$N_MEM" $EXTRA_DATA_ARGS)

# ------------------ 1) detección YOLOX + trackers + eventos ------------------
if has_stage mine; then
  for split in train val; do
    missing=0
    for t in $TRACKERS; do [[ -s "$EV/$split/events_$t.jsonl" ]] || missing=1; done
    if (( missing )); then
      log "Minando eventos: $split"
      # shellcheck disable=SC2086
      python mine_events.py --data_root "$DT_ROOT/$split" --det_dir "$DETS/$split" \
        --trackers $TRACKERS --out_dir "$EV/$split" --fp16 $([[ "$REID" == 1 ]] && echo --reid)
    else
      log "Eventos de $split ya existen, se omite"
    fi
  done
  # val se parte POR SECUENCIA: "sel" para elegir checkpoint/beta, "test" para reportar.
  # Si se reporta sobre lo mismo que se usó para seleccionar, el resultado sale optimista.
  python - <<EOF
import random
from pathlib import Path
from common import read_jsonl, write_jsonl
for f in Path("$EV/val").glob("events_*.jsonl"):
    if f.stem.endswith((".sel", ".test")):
        continue
    evs = read_jsonl(f)
    seqs = sorted({e["seq"] for e in evs})
    random.Random(0).shuffle(seqs)
    sel = set(seqs[: len(seqs) // 2])
    write_jsonl(f.with_name(f.stem + ".sel.jsonl"), [e for e in evs if e["seq"] in sel])
    write_jsonl(f.with_name(f.stem + ".test.jsonl"), [e for e in evs if e["seq"] not in sel])
    print(f.name, "secuencias sel/test:", len(sel), len(seqs) - len(sel))
EOF
fi

# --------------------------- 2) prueba congelada ----------------------------
if has_stage zeroshot; then
  out="$WORK_DIR/runs/zs_$RUN_TAG"
  reuse=(); [[ -s "$out/scores.jsonl" ]] && reuse=(--reuse_scores)
  log "Zero-shot -> $out"
  "${LAUNCH[@]}" evaluate.py --model "$MODEL" --dtype "$DTYPE" --events "$EV/val/events_$VAL_TRACKER.test.jsonl" \
    --out "$out" "${DATA_ARGS[@]}" "${reuse[@]}"
fi

# ----------------------------- 3) fine-tuning -------------------------------
if has_stage train; then
  out="$WORK_DIR/runs/lora_$RUN_TAG"
  log "LoRA -> $out"
  "${LAUNCH[@]}" train_lora.py --model "$MODEL" --dtype "$DTYPE" \
    --train_events "$EV"/train/events_*.jsonl --val_events "$EV/val/events_$VAL_TRACKER.sel.jsonl" \
    --out "$out" --resume --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" "${DATA_ARGS[@]}"
fi

# ------------- 4) evaluación final del mejor adaptador (todos los trackers) -------------
if has_stage eval; then
  best="$WORK_DIR/runs/lora_$RUN_TAG/best"
  if [[ -d "$best" ]]; then
    for t in $TRACKERS; do
      log "Evaluación final ($t)"
      "${LAUNCH[@]}" evaluate.py --model "$MODEL" --dtype "$DTYPE" --adapter "$best" \
        --events "$EV/val/events_$t.test.jsonl" --out "$WORK_DIR/runs/final_${RUN_TAG}_$t" "${DATA_ARGS[@]}"
    done
  else
    log "No hay adaptador guardado (el fine-tuning no superó al zero-shot): se omite la evaluación final"
  fi
fi

log "Terminado"

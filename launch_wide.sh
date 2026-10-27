#!/bin/bash
# Entrenamiento amplio: un job por versión del modelo, más épocas, checkpoints de cada evaluación y parada temprana.
# Corre en el nodo login (lanza los jobs con sbatch).
#
#   DT_ROOT=/ruta/DanceTrack bash launch_wide.sh
#   DT_ROOT=... MODELS="qwen3vl-2b qwen3vl-4b qwen3vl-8b" EPOCHS=8 PATIENCE=6 CHAIN=1 bash launch_wide.sh
#
# Variables (todas opcionales salvo DT_ROOT):
#   MODELS   modelos, uno por job            (def. "qwen3vl-2b qwen3vl-4b qwen3vl-8b"; con DDP cada GPU carga el modelo entero)
#   EPOCHS   épocas por modelo               (def. 8)
#   PATIENCE evaluaciones sin mejorar el neto antes de parar (def. 6 = 3 épocas con 2 evals/época; 0 = nunca)
#   LR, BATCH, VIEW_MODE, N_MEM, REID        se pasan tal cual al .slurm
#   HARD_OVERSAMPLE=3  repite los pares donde el tracker erró;  AUGMENT=1  aumentación de imagen en train
#   MINE_VARIANTS="ms03+ms005+ti01"  minado extra de train con otros umbrales (separadas por "+")
#   CHAIN    1 = un job tras otro (afterany; cada uno usa todas las GPUs); 0 = todos a la vez (def. 1)
# Resultados: work/runs/lora_<modelo>_<vista>_m<N>_e<EPOCHS>/{best,last,ckpt_ep*_it*,history.jsonl}
# Si un job se corta (límite de tiempo), vuelve a lanzar el mismo comando: --resume continúa desde last/.
set -euo pipefail
DT_ROOT="$(realpath "${DT_ROOT:?define DT_ROOT}")"
MODELS="${MODELS:-qwen3vl-2b qwen3vl-4b qwen3vl-8b}"
EPOCHS="${EPOCHS:-8}"
CHAIN="${CHAIN:-1}"
cd "$(dirname "$0")"
mkdir -p logs

passthru="PROFILE=wide,DT_ROOT=$DT_ROOT,EPOCHS=$EPOCHS"
for v in PATIENCE LR BATCH VIEW_MODE N_MEM MEM_STRIDE REID HARD_OVERSAMPLE AUGMENT MINE_VARIANTS; do
  [[ -n "${!v:-}" ]] && passthru+=",$v=${!v}"
done

prev=""
for m in $MODELS; do
  dep=()
  [[ "$CHAIN" == 1 && -n "$prev" ]] && dep=(--dependency="afterany:$prev")
  jid="$(sbatch --parsable --job-name="wide_$m" "${dep[@]}" --export="NONE,MODEL=$m,$passthru" run_vlm_xing.slurm)"
  echo "$m -> job $jid ${dep[*]:-}"
  prev="$jid"
done
echo "Seguimiento: squeue -u \$USER ; curvas: python plot_history.py work/runs/lora_*_e$EPOCHS"

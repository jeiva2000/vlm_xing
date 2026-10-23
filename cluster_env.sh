# Módulos y variables CUDA/GCC del cluster. Uso: source cluster_env.sh
# Variables: MODULES (def. "GCC/12.2.0 CUDA/11.8.0"), PYTHON_MODULE (opcional)
if command -v module >/dev/null 2>&1; then
  module purge || true
  # shellcheck disable=SC2086
  # se carga módulo a módulo: si uno falla (p.ej. GCC con otro nombre en este nodo) se avisa y se sigue;
  # lo imprescindible es nvcc (se comprueba abajo)
  _failed=0
  for _m in ${MODULES:-GCC/12.2.0 CUDA/11.8.0} ${PYTHON_MODULE:-}; do
    module load "$_m" || { echo "AVISO: no se pudo cargar el módulo $_m en $(hostname)" >&2; _failed=1; }
  done
  if (( _failed )) && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "       (login: se continúa) Nombres disponibles: 'module avail CUDA' / 'module avail GCC'; usa MODULES=\"...\"" >&2
    return 0 2>/dev/null || exit 0
  fi

  if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc no está en PATH tras cargar los módulos (${MODULES:-GCC/12.2.0 CUDA/11.8.0}). Revisa 'module avail CUDA' en este nodo ($(hostname))." >&2
    module avail CUDA 2>&1 >&2
    return 1 2>/dev/null || exit 1
  fi
  CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  export CUDA_HOME
  export CUDA_PATH="$CUDA_HOME"
  CC="$(which gcc)"; CXX="$(which g++)"
  export CC CXX
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6}"   # A30/A100 (8.0) y A40 (8.6)
fi
#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export NCCL_ASYNC_ERROR_HANDLING=1

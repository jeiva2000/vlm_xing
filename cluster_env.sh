# Módulos y variables CUDA/GCC del cluster. Uso: source cluster_env.sh
# Variables: MODULES (def. "GCC/12.2.0 CUDA/11.8.0"), PYTHON_MODULE (opcional)
if command -v module >/dev/null 2>&1; then
  module purge || true
  # shellcheck disable=SC2086
  if ! module load ${MODULES:-GCC/12.2.0 CUDA/11.8.0} ${PYTHON_MODULE:-}; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
      # el nodo login puede no tener los módulos de los nodos GPU: se sigue sin ellos
      echo "AVISO: no se pudieron cargar los módulos (${MODULES:-GCC/12.2.0 CUDA/11.8.0}) en $(hostname); se continúa sin CUDA/GCC." >&2
      echo "       Nombres disponibles: 'module avail CUDA' / 'module avail GCC'; usa MODULES=\"...\" si difieren." >&2
      return 0 2>/dev/null || exit 0
    fi
    echo "ERROR: fallo al cargar los módulos (${MODULES:-GCC/12.2.0 CUDA/11.8.0}) en $(hostname)" >&2
    module avail 2>&1 >&2
    return 1 2>/dev/null || exit 1
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

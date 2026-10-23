# Crea el entorno si no existe (conda, o venv si no hay conda) e instala dependencias; si existe, lo activa.
# Uso: source setup_env.sh   (desde la carpeta del código)
# Variables: CONDA_ENV, CONDA_SH, TORCH_SPEC, TORCH_INDEX, FORCE_INSTALL=1 (reinstala en un entorno existente)
# La creación/instalación necesita internet en el nodo donde corra.
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.7.1 torchvision==0.22.1}"      # última serie con wheels cu118
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"
_log() { echo "[$(date '+%F %T')] $*"; }

_setup_env() {
  CREATE_ENV=1 source "$CODE_DIR/activate_env.sh" || return 1
  local marker="$ENV_PREFIX/.deps_installed"
  if [[ -f "$marker" && "${FORCE_INSTALL:-0}" != 1 ]]; then
    _log "Entorno existente, solo se activa: $ENV_PREFIX"
    return
  fi
  python -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
    || { echo "Se necesita Python >= 3.10 en el entorno"; return 1; }
  _log "Instalando dependencias en $ENV_PREFIX"
  pip install -U pip wheel setuptools
  # shellcheck disable=SC2086
  pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
  pip install -r "$CODE_DIR/requirements.txt"
  pip install --no-build-isolation --no-deps yolox==0.3.0
  touch "$marker"
}

# flock evita que dos jobs creen/instalen a la vez sobre el mismo entorno.
# Se ejecuta en el shell actual (no en subshell) para que la activación persista.
mkdir -p "$HOME/envs"
exec 9>"$HOME/envs/${CONDA_ENV:-vlm_xing}.lock"
flock -w 3600 9 || { echo "No se pudo bloquear el entorno"; exit 1; }
_setup_env
_rc=$?
flock -u 9; exec 9>&-
[[ $_rc -eq 0 ]] || { echo "Falló la preparación del entorno"; return $_rc 2>/dev/null || exit $_rc; }

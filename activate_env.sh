# Activa el entorno del proyecto: conda si se encuentra conda.sh (miniconda/anaconda), si no un venv.
# Uso: source activate_env.sh            Variables: CONDA_ENV (def. vlm_xing), CONDA_SH, ENV_DIR (venv)
# Con CREATE_ENV=1 crea el entorno conda si no existe (solo en el nodo login, necesita internet).
CONDA_ENV="${CONDA_ENV:-vlm_xing}"
_conda_sh=""
for _c in "${CONDA_SH:-}" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" \
          "$HOME/.conda/etc/profile.d/conda.sh"; do
  [[ -n "$_c" && -f "$_c" ]] && { _conda_sh="$_c"; break; }
done
if [[ -n "$_conda_sh" ]]; then
  _had_u=0; [[ $- == *u* ]] && _had_u=1
  set +u  # los scripts de (de)activación de conda usan variables sin definir
  # shellcheck disable=SC1090
  source "$_conda_sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    if [[ "${CREATE_ENV:-0}" == 1 ]]; then
      conda create -y -n "$CONDA_ENV" "python=${CONDA_PYTHON:-3.11}" pip
    else
      echo "No existe el entorno conda '$CONDA_ENV' (se crea al lanzar run_vlm_xing.slurm)"; return 1 2>/dev/null || exit 1
    fi
  fi
  conda activate "$CONDA_ENV"
  (( _had_u )) && set -u
  ENV_PREFIX="$CONDA_PREFIX"
else
  ENV_DIR="${ENV_DIR:-$HOME/envs/vlm_xing}"
  if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
    if [[ "${CREATE_ENV:-0}" == 1 ]]; then
      "${PYTHON_BIN:-python3}" -m venv "$ENV_DIR"
    else
      echo "No existe el venv $ENV_DIR ni se encontró conda.sh (define CONDA_SH)"; return 1 2>/dev/null || exit 1
    fi
  fi
  # shellcheck disable=SC1091
  source "$ENV_DIR/bin/activate"
  ENV_PREFIX="$ENV_DIR"
fi

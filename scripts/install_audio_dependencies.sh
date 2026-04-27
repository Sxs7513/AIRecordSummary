#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-audio"
PIP_CONSTRAINTS_FILE="${ROOT_DIR}/scripts/audio-python-constraints.txt"
PYTHON_BIN_DEFAULT="python3.10"
PYTHON_VERSION_MIN="3.10"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"

log() {
  printf '[install-audio-deps] %s\n' "$1"
}

warn() {
  printf '[install-audio-deps][warn] %s\n' "$1" >&2
}

fail() {
  printf '[install-audio-deps][error] %s\n' "$1" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

brew_install() {
  local package_name="$1"
  local output_file
  output_file="$(mktemp)"
  if brew install "${package_name}" 2>&1 | tee "${output_file}"; then
    rm -f "${output_file}"
    return 0
  fi

  if grep -q "unknown or unsupported macOS version" "${output_file}"; then
    rm -f "${output_file}"
    fail "Homebrew is too old for this macOS version. Run 'brew update' first, then rerun this script."
  fi

  rm -f "${output_file}"
  return 1
}

python_meets_min_version() {
  local python_bin="$1"
  "${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

find_supported_python() {
  local python_bin
  for python_bin in python3.10 python3; do
    if command_exists "${python_bin}" && python_meets_min_version "${python_bin}"; then
      printf '%s\n' "${python_bin}"
      return 0
    fi
  done

  return 1
}

brew_services_available() {
  command_exists brew && brew help services >/dev/null 2>&1
}

find_pg_ctl() {
  if command_exists pg_ctl; then
    command -v pg_ctl
    return 0
  fi

  local candidate
  for candidate in \
    /usr/local/opt/postgresql@16/bin/pg_ctl \
    /usr/local/opt/postgresql@15/bin/pg_ctl \
    /usr/local/opt/postgresql@14/bin/pg_ctl \
    /usr/local/opt/postgresql/bin/pg_ctl \
    /opt/homebrew/opt/postgresql@16/bin/pg_ctl \
    /opt/homebrew/opt/postgresql@15/bin/pg_ctl \
    /opt/homebrew/opt/postgresql@14/bin/pg_ctl \
    /opt/homebrew/opt/postgresql/bin/pg_ctl; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

start_postgresql_with_pg_ctl() {
  local pg_ctl_bin="$1"
  local data_dir="$2"

  if [[ ! -x "${pg_ctl_bin}" || ! -d "${data_dir}" ]]; then
    return 1
  fi

  log "Starting PostgreSQL via pg_ctl (${data_dir})."
  if "${pg_ctl_bin}" -D "${data_dir}" start >/dev/null; then
    return 0
  fi

  warn "pg_ctl could not start PostgreSQL from ${data_dir}; trying the next known location."
  return 1
}

load_env_file() {
  local env_file="${ROOT_DIR}/.env"
  if [[ ! -f "${env_file}" ]]; then
    return
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]]; then
      continue
    fi
    local key="${line%%=*}"
    local value="${line#*=}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    if [[ -z "${!key+x}" ]]; then
      export "${key}=${value}"
    fi
  done <"${env_file}"

  LLM_CORRECTION_ENABLED="${LLM_CORRECTION_ENABLED:-false}"
  LLM_CORRECTION_MODEL_REPO="${LLM_CORRECTION_MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct-GGUF}"
  LLM_CORRECTION_MODEL_FILE="${LLM_CORRECTION_MODEL_FILE:-qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf,qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf}"
  AUDIO_MODEL_CACHE_ROOT="${AUDIO_MODEL_CACHE_ROOT:-model-cache}"
  ASR_PROVIDER="${ASR_PROVIDER:-whisper}"
  EMBEDDING_ENABLED="${EMBEDDING_ENABLED:-true}"
  EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
  EMBEDDING_MODEL_CACHE_DIR="${EMBEDDING_MODEL_CACHE_DIR:-model-cache/embedding}"
  RAG_ANSWER_ENABLED="${RAG_ANSWER_ENABLED:-true}"
  RAG_ANSWER_PROVIDER="${RAG_ANSWER_PROVIDER:-local_llm}"
  RAG_ANSWER_MODEL_REPO="${RAG_ANSWER_MODEL_REPO:-Qwen/Qwen3-8B-GGUF}"
  RAG_ANSWER_MODEL_FILE="${RAG_ANSWER_MODEL_FILE:-Qwen3-8B-Q4_K_M.gguf}"
  PGVECTOR_VERSION="${PGVECTOR_VERSION:-0.8.2}"
}

pip_install() {
  "${VENV_PIP}" install --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" -c "${PIP_CONSTRAINTS_FILE}" "$@"
}

pip_install_with_pypi_fallback() {
  if pip_install "$@"; then
    return
  fi
  warn "Install from ${PIP_INDEX_URL} failed, retrying with official PyPI."
  "${VENV_PIP}" install --index-url "https://pypi.org/simple" -c "${PIP_CONSTRAINTS_FILE}" "$@"
}

pip_install_binary_from_pypi() {
  "${VENV_PIP}" install --index-url "https://pypi.org/simple" --only-binary=:all: -c "${PIP_CONSTRAINTS_FILE}" "$@"
}

python_pip_install() {
  "${VENV_PYTHON}" -m pip install --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" -c "${PIP_CONSTRAINTS_FILE}" "$@"
}

detect_platform() {
  case "$(uname -s)" in
    Darwin) echo "macos" ;;
    Linux) echo "linux" ;;
    *) echo "unknown" ;;
  esac
}

ensure_python() {
  PYTHON_BIN="$(find_supported_python || true)"
  if [[ -n "${PYTHON_BIN}" ]]; then
    log "Found Python: ${PYTHON_BIN}"
    return
  fi

  local platform
  platform="$(detect_platform)"

  if [[ "${platform}" == "macos" ]] && command_exists brew; then
    log "Python ${PYTHON_VERSION_MIN}+ not found. Installing python@3.10 via Homebrew."
    brew_install python@3.10
    PYTHON_BIN="$(find_supported_python || true)"
    if [[ -n "${PYTHON_BIN}" ]]; then
      return
    fi
    fail "Installed python@3.10, but no Python ${PYTHON_VERSION_MIN}+ binary was found in PATH."
  fi

  if [[ "${platform}" == "linux" ]]; then
    if command_exists apt-get; then
      log "Python ${PYTHON_VERSION_MIN}+ not found. Installing python3.10 via apt-get."
      sudo apt-get update
      sudo apt-get install -y python3.10 python3.10-venv python3-pip
      PYTHON_BIN="$(find_supported_python || true)"
      if [[ -n "${PYTHON_BIN}" ]]; then
        return
      fi
      fail "Installed python3.10, but no Python ${PYTHON_VERSION_MIN}+ binary was found in PATH."
    fi

    if command_exists dnf; then
      log "Python ${PYTHON_VERSION_MIN}+ not found. Installing python3.10 via dnf."
      sudo dnf install -y python3.10 python3.10-pip
      PYTHON_BIN="$(find_supported_python || true)"
      if [[ -n "${PYTHON_BIN}" ]]; then
        return
      fi
      fail "Installed python3.10, but no Python ${PYTHON_VERSION_MIN}+ binary was found in PATH."
    fi

    if command_exists yum; then
      log "Python ${PYTHON_VERSION_MIN}+ not found. Installing python3.10 via yum."
      sudo yum install -y python3.10 python3.10-pip
      PYTHON_BIN="$(find_supported_python || true)"
      if [[ -n "${PYTHON_BIN}" ]]; then
        return
      fi
      fail "Installed python3.10, but no Python ${PYTHON_VERSION_MIN}+ binary was found in PATH."
    fi
  fi

  fail "Python ${PYTHON_VERSION_MIN}+ is required but no supported installer was found."
}

ensure_build_tools() {
  local platform
  platform="$(detect_platform)"

  if command_exists cmake; then
    log "cmake already installed."
  else
    if [[ "${platform}" == "macos" ]] && command_exists brew; then
      log "cmake not found. Installing via Homebrew."
      brew_install cmake
    elif [[ "${platform}" == "linux" ]]; then
      if command_exists apt-get; then
        log "cmake not found. Installing via apt-get."
        sudo apt-get update
        sudo apt-get install -y cmake build-essential
      elif command_exists dnf; then
        log "cmake not found. Installing via dnf."
        sudo dnf install -y cmake gcc gcc-c++ make
      elif command_exists yum; then
        log "cmake not found. Installing via yum."
        sudo yum install -y cmake gcc gcc-c++ make
      else
        fail "cmake is required to build llvmlite, but no supported installer was found."
      fi
    else
      fail "cmake is required to build llvmlite, but no supported installer was found."
    fi
  fi

  if [[ "${platform}" == "macos" ]] && command_exists brew; then
    local llvm_prefix
    if brew list llvm >/dev/null 2>&1; then
      log "llvm already installed."
    else
      log "llvm not found. Installing via Homebrew."
      brew_install llvm
    fi

    llvm_prefix="$(brew --prefix llvm)"
    export LLVM_DIR="${llvm_prefix}/lib/cmake/llvm"
    export LLVM_CONFIG="${llvm_prefix}/bin/llvm-config"
    export CMAKE_PREFIX_PATH="${llvm_prefix}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
    log "Configured LLVM_DIR=${LLVM_DIR}"
    return
  fi

  if [[ "${platform}" == "linux" ]]; then
    if command_exists apt-get; then
      log "Installing LLVM development package via apt-get."
      sudo apt-get update
      sudo apt-get install -y llvm-dev
      return
    fi

    if command_exists dnf; then
      log "Installing LLVM development package via dnf."
      sudo dnf install -y llvm-devel
      return
    fi

    if command_exists yum; then
      log "Installing LLVM development package via yum."
      sudo yum install -y llvm-devel
      return
    fi
  fi

  fail "LLVM development files are required to build llvmlite, but no supported installer was found."
}

start_postgresql_service() {
  local platform
  platform="$(detect_platform)"

  if [[ "${platform}" == "macos" ]]; then
    if brew_services_available; then
      local formula
      for formula in postgresql@16 postgresql@15 postgresql@14 postgresql; do
        if brew list "${formula}" >/dev/null 2>&1; then
          log "Starting PostgreSQL service via Homebrew (${formula})."
          if brew services start "${formula}" >/dev/null; then
            return 0
          fi
          warn "Homebrew service start failed for ${formula}; trying direct pg_ctl startup."
        fi
      done
    fi

    start_postgresql_with_pg_ctl /usr/local/opt/postgresql@16/bin/pg_ctl /usr/local/var/postgresql@16 && return 0
    start_postgresql_with_pg_ctl /usr/local/opt/postgresql@15/bin/pg_ctl /usr/local/var/postgresql@15 && return 0
    start_postgresql_with_pg_ctl /usr/local/opt/postgresql@14/bin/pg_ctl /usr/local/var/postgresql@14 && return 0
    start_postgresql_with_pg_ctl /usr/local/opt/postgresql/bin/pg_ctl /usr/local/var/postgres && return 0
    start_postgresql_with_pg_ctl /opt/homebrew/opt/postgresql@16/bin/pg_ctl /opt/homebrew/var/postgresql@16 && return 0
    start_postgresql_with_pg_ctl /opt/homebrew/opt/postgresql@15/bin/pg_ctl /opt/homebrew/var/postgresql@15 && return 0
    start_postgresql_with_pg_ctl /opt/homebrew/opt/postgresql@14/bin/pg_ctl /opt/homebrew/var/postgresql@14 && return 0
    start_postgresql_with_pg_ctl /opt/homebrew/opt/postgresql/bin/pg_ctl /opt/homebrew/var/postgres && return 0

    local pg_ctl_bin=""
    pg_ctl_bin="$(find_pg_ctl || true)"
    if [[ -n "${pg_ctl_bin}" ]]; then
      start_postgresql_with_pg_ctl "${pg_ctl_bin}" /usr/local/var/postgresql@16 && return 0
      start_postgresql_with_pg_ctl "${pg_ctl_bin}" /usr/local/var/postgresql@15 && return 0
      start_postgresql_with_pg_ctl "${pg_ctl_bin}" /usr/local/var/postgresql@14 && return 0
      start_postgresql_with_pg_ctl "${pg_ctl_bin}" /usr/local/var/postgres && return 0
      start_postgresql_with_pg_ctl "${pg_ctl_bin}" /opt/homebrew/var/postgresql@16 && return 0
      start_postgresql_with_pg_ctl "${pg_ctl_bin}" /opt/homebrew/var/postgresql@15 && return 0
      start_postgresql_with_pg_ctl "${pg_ctl_bin}" /opt/homebrew/var/postgresql@14 && return 0
      start_postgresql_with_pg_ctl "${pg_ctl_bin}" /opt/homebrew/var/postgres && return 0
    fi

    warn "PostgreSQL service start was skipped because no supported macOS service manager was found."
    return 0
  fi

  if [[ "${platform}" == "linux" ]]; then
    if command_exists systemctl; then
      local service_name
      for service_name in postgresql postgresql-16 postgresql-15 postgresql-14 postgresql-13 postgresql-12; do
        if systemctl list-unit-files "${service_name}.service" >/dev/null 2>&1; then
          log "Starting PostgreSQL service via systemctl (${service_name})."
          sudo systemctl enable --now "${service_name}" >/dev/null || true
          return 0
        fi
      done
    fi

    if command_exists pg_lsclusters && command_exists pg_ctlcluster; then
      local cluster_version
      local cluster_name
      cluster_version="$(pg_lsclusters --no-header 2>/dev/null | awk 'NR == 1 {print $1}')"
      cluster_name="$(pg_lsclusters --no-header 2>/dev/null | awk 'NR == 1 {print $2}')"
      if [[ -n "${cluster_version}" && -n "${cluster_name}" ]]; then
        log "Starting PostgreSQL cluster via pg_ctlcluster (${cluster_version}/${cluster_name})."
        sudo pg_ctlcluster "${cluster_version}" "${cluster_name}" start >/dev/null || true
        return 0
      fi
    fi

    if command_exists service; then
      log "Starting PostgreSQL service via service."
      sudo service postgresql start >/dev/null || true
      return 0
    fi

    if command_exists pg_ctl && [[ -d /var/lib/pgsql/data ]]; then
      log "Starting PostgreSQL via pg_ctl."
      sudo -u postgres pg_ctl -D /var/lib/pgsql/data start >/dev/null || true
      return 0
    fi

    warn "PostgreSQL service start was skipped because no supported Linux service manager was found."
    return 0
  fi

  warn "PostgreSQL service start was skipped on unsupported platform: ${platform}."
}

ensure_postgresql() {
  local platform
  platform="$(detect_platform)"

  if command_exists pg_isready; then
    log "PostgreSQL client tools already installed."
  elif command_exists psql; then
    log "PostgreSQL client already installed."
  else
    if [[ "${platform}" == "macos" ]] && command_exists brew; then
      log "PostgreSQL not found. Installing PostgreSQL via Homebrew."
      brew_install postgresql@16
    elif [[ "${platform}" == "linux" ]]; then
      if command_exists apt-get; then
        log "PostgreSQL not found. Installing PostgreSQL via apt-get."
        sudo apt-get update
        sudo apt-get install -y postgresql postgresql-client
      elif command_exists dnf; then
        log "PostgreSQL not found. Installing PostgreSQL via dnf."
        sudo dnf install -y postgresql-server postgresql
        if [[ ! -d /var/lib/pgsql/data/base ]]; then
          sudo postgresql-setup --initdb
        fi
      elif command_exists yum; then
        log "PostgreSQL not found. Installing PostgreSQL via yum."
        sudo yum install -y postgresql-server postgresql
        if [[ ! -d /var/lib/pgsql/data/base ]]; then
          sudo postgresql-setup initdb
        fi
      else
        fail "PostgreSQL is required but no supported installer was found."
      fi
    else
      fail "PostgreSQL is required but no supported installer was found."
    fi
  fi

  start_postgresql_service

  if command_exists pg_isready; then
    local db_host="${DB_HOST:-localhost}"
    local db_port="${DB_PORT:-5432}"
    if pg_isready -h "${db_host}" -p "${db_port}" >/dev/null 2>&1; then
      log "PostgreSQL is accepting connections at ${db_host}:${db_port}."
    else
      warn "PostgreSQL was installed, but ${db_host}:${db_port} is not accepting connections yet."
      warn "Check the service status and make sure .env DB_HOST/DB_PORT match your local PostgreSQL instance."
    fi
  fi
}

psql_available_extension() {
  local extension_name="$1"
  if ! command_exists psql; then
    return 1
  fi

  local db_host="${DB_HOST:-localhost}"
  local db_port="${DB_PORT:-5432}"
  local db_user="${DB_USER:-postgres}"
  local db_password="${DB_PASSWORD:-}"
  local db_admin_database="${DB_ADMIN_DATABASE:-postgres}"
  local result
  result="$(PGPASSWORD="${db_password}" psql -h "${db_host}" -p "${db_port}" -U "${db_user}" -d "${db_admin_database}" -tAc "select 1 from pg_available_extensions where name = '${extension_name}'" 2>/dev/null || true)"
  [[ "${result}" == "1" ]]
}

pgvector_control_available() {
  if ! command_exists pg_config; then
    return 1
  fi

  local sharedir
  sharedir="$(pg_config --sharedir 2>/dev/null || true)"
  [[ -n "${sharedir}" && -f "${sharedir}/extension/vector.control" ]]
}

install_pgvector_from_source() {
  if ! command_exists pg_config; then
    fail "pg_config is required to build pgvector for the active PostgreSQL installation."
  fi
  if ! command_exists make; then
    fail "make is required to build pgvector."
  fi
  if ! command_exists curl; then
    fail "curl is required to download pgvector source."
  fi

  local work_dir
  work_dir="$(mktemp -d)"
  local archive_path="${work_dir}/pgvector.tar.gz"
  local source_url="https://github.com/pgvector/pgvector/archive/refs/tags/v${PGVECTOR_VERSION}.tar.gz"
  log "Downloading pgvector ${PGVECTOR_VERSION} source for current pg_config: $(pg_config --version)."
  curl -fsSL "${source_url}" -o "${archive_path}"
  tar -xzf "${archive_path}" -C "${work_dir}"

  local source_dir="${work_dir}/pgvector-${PGVECTOR_VERSION}"
  if [[ ! -d "${source_dir}" ]]; then
    fail "Downloaded pgvector source archive did not contain ${source_dir}."
  fi

  log "Building pgvector against $(pg_config --bindir 2>/dev/null || echo pg_config)."
  make -C "${source_dir}"

  local sharedir
  sharedir="$(pg_config --sharedir)"
  if [[ -w "${sharedir}/extension" ]]; then
    make -C "${source_dir}" install
  else
    log "Installing pgvector with sudo because ${sharedir}/extension is not writable."
    sudo make -C "${source_dir}" install
  fi
  rm -rf "${work_dir}"
}

ensure_pgvector() {
  if [[ "${EMBEDDING_ENABLED:-true}" != "true" ]]; then
    log "Skipping pgvector setup because EMBEDDING_ENABLED is not true."
    return
  fi

  if psql_available_extension vector; then
    log "pgvector extension is available."
    return
  fi

  if pgvector_control_available; then
    log "pgvector control file exists for active pg_config; PostgreSQL may need a service restart."
    start_postgresql_service
    if psql_available_extension vector; then
      log "pgvector extension is available after PostgreSQL restart."
      return
    fi
  fi

  local platform
  platform="$(detect_platform)"

  if [[ "${platform}" == "macos" ]] && command_exists brew; then
    log "pgvector extension not found. Installing pgvector via Homebrew."
    brew_install pgvector || warn "Homebrew could not install pgvector automatically."
    if ! pgvector_control_available; then
      warn "Homebrew pgvector is not installed for active $(pg_config --version 2>/dev/null || echo PostgreSQL). Building pgvector from source."
      install_pgvector_from_source
    fi
  elif [[ "${platform}" == "linux" ]]; then
    if command_exists apt-get; then
      local pg_major
      pg_major="$(pg_config --version 2>/dev/null | sed -E 's/^PostgreSQL ([0-9]+).*/\1/' || true)"
      if [[ -n "${pg_major}" ]]; then
        log "pgvector extension not found. Trying apt package postgresql-${pg_major}-pgvector."
        sudo apt-get update
        sudo apt-get install -y "postgresql-${pg_major}-pgvector" || warn "apt could not install postgresql-${pg_major}-pgvector automatically."
      else
        warn "pg_config not found; cannot infer PostgreSQL major version for pgvector apt package."
      fi
    elif command_exists dnf; then
      log "pgvector extension not found. Trying dnf package pgvector."
      sudo dnf install -y pgvector || warn "dnf could not install pgvector automatically."
    elif command_exists yum; then
      log "pgvector extension not found. Trying yum package pgvector."
      sudo yum install -y pgvector || warn "yum could not install pgvector automatically."
    else
      warn "pgvector extension is required, but no supported installer was found."
    fi
  else
    warn "pgvector extension is required, but this platform is unsupported for automatic installation."
  fi

  if ! pgvector_control_available; then
    warn "pgvector control file is still missing for active $(pg_config --version 2>/dev/null || echo PostgreSQL). Trying source build."
    install_pgvector_from_source
  fi

  start_postgresql_service

  if psql_available_extension vector; then
    log "pgvector extension is available after installation."
  else
    fail "pgvector is still not visible to PostgreSQL. Install pgvector for the active PostgreSQL server before starting the app."
  fi
}

ensure_ffmpeg() {
  if command_exists ffmpeg; then
    log "ffmpeg already installed."
    return
  fi

  if [[ -x "${VENV_DIR}/bin/ffmpeg" ]]; then
    log "ffmpeg wrapper already exists in virtual environment."
    return
  fi

  if ! python_has_module imageio_ffmpeg; then
    log "Installing imageio-ffmpeg in virtual environment via ${PIP_INDEX_URL}."
    pip_install imageio-ffmpeg
  fi

  log "Creating ffmpeg wrapper in virtual environment."
cat >"${VENV_DIR}/bin/ffmpeg" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/python" -c 'import imageio_ffmpeg, os, sys; os.execv(imageio_ffmpeg.get_ffmpeg_exe(), ["ffmpeg", *sys.argv[1:]])' "$@"
EOF
  chmod +x "${VENV_DIR}/bin/ffmpeg"
}

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtual environment at ${VENV_DIR}."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    VENV_CREATED=1
  else
    log "Virtual environment already exists at ${VENV_DIR}."
    VENV_CREATED=0
  fi

  VENV_PYTHON="${VENV_DIR}/bin/python"
  VENV_PIP="${VENV_DIR}/bin/pip"

  if [[ ! -x "${VENV_PYTHON}" ]]; then
    fail "Virtual environment python binary not found at ${VENV_PYTHON}."
  fi

  local venv_python_version
  venv_python_version="$("${VENV_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if ! python_meets_min_version "${VENV_PYTHON}"; then
    fail "Existing virtual environment uses Python ${venv_python_version}, but this project requires Python ${PYTHON_VERSION_MIN} or newer. Remove ${VENV_DIR} and rerun this script to recreate it with ${PYTHON_BIN_DEFAULT}."
  fi
}

upgrade_base_tools() {
  log "Upgrading pip, setuptools, and wheel in virtual environment via ${PIP_INDEX_URL}."
  python_pip_install --upgrade pip setuptools wheel
}

python_has_module() {
  local module_name="$1"
  "${VENV_PYTHON}" -c "import importlib; importlib.import_module('${module_name}')" >/dev/null 2>&1
}

ensure_torch() {
  if python_has_module torch; then
    log "torch already installed."
    return
  fi

  log "Installing torch via ${PIP_INDEX_URL}."
  pip_install torch
}

ensure_whisper() {
  if python_has_module whisper; then
    log "whisper already installed."
    return
  fi

  log "Installing openai-whisper from official GitHub repository. Python dependencies use ${PIP_INDEX_URL}."
  pip_install "git+https://github.com/openai/whisper.git"
}

ensure_numba_runtime() {
  if python_has_module llvmlite && python_has_module numba && "${VENV_PYTHON}" -c 'import numpy; raise SystemExit(0 if int(numpy.__version__.split(".")[0]) < 2 else 1)' >/dev/null 2>&1; then
    log "numpy, llvmlite, and numba already installed with compatible versions."
    return
  fi

  log "Installing compatible numpy, llvmlite, and numba binary wheels from official PyPI."
  pip_install_binary_from_pypi --force-reinstall numpy llvmlite numba certifi
}

ensure_funasr() {
  if [[ "${ASR_PROVIDER}" != "sensevoice" && "${ASR_PROVIDER}" != "paraformer" && "${ASR_PROVIDER}" != "hf_whisper" && "${ASR_PROVIDER}" != "qwen_asr" ]]; then
    log "Skipping FunASR setup because ASR_PROVIDER is not sensevoice, paraformer, hf_whisper, or qwen_asr."
    return
  fi

  if python_has_module funasr; then
    log "funasr already installed."
  else
    log "Installing FunASR via ${PIP_INDEX_URL}."
    pip_install funasr modelscope huggingface_hub
  fi
}

ensure_transformers_asr() {
  if [[ "${ASR_PROVIDER}" != "hf_whisper" ]]; then
    log "Skipping Transformers ASR setup because ASR_PROVIDER is not hf_whisper."
    return
  fi

  if python_has_module transformers; then
    log "transformers already installed."
  else
    log "Installing Transformers ASR dependencies via ${PIP_INDEX_URL}."
    pip_install transformers accelerate huggingface_hub soundfile librosa
  fi
}

ensure_qwen_asr() {
  if [[ "${ASR_PROVIDER}" != "qwen_asr" ]]; then
    log "Skipping Qwen3-ASR setup because ASR_PROVIDER is not qwen_asr."
    return
  fi

  local python_version
  python_version="$("${VENV_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if ! python_meets_min_version "${VENV_PYTHON}"; then
    fail "Qwen3-ASR requires Python ${PYTHON_VERSION_MIN} or newer; current virtual environment uses Python ${python_version}."
  fi

  if python_has_module qwen_asr; then
    log "qwen-asr already installed."
  else
    log "Installing Qwen3-ASR runtime via ${PIP_INDEX_URL}."
    pip_install_with_pypi_fallback qwen-asr
  fi
}

ensure_pyannote() {
  if python_has_module pyannote.audio; then
    log "pyannote.audio already installed."
    return
  fi

  log "Installing pyannote.audio via ${PIP_INDEX_URL}."
  pip_install pyannote.audio
}

ensure_speechbrain() {
  if python_has_module speechbrain; then
    log "speechbrain already installed."
    return
  fi

  log "Installing SpeechBrain via ${PIP_INDEX_URL}."
  pip_install speechbrain
}

ensure_pycorrector() {
  if python_has_module pycorrector; then
    log "pycorrector already installed."
  else
    log "Installing pycorrector via ${PIP_INDEX_URL}."
    pip_install pycorrector
  fi
}

ensure_embedding_runtime() {
  if [[ "${EMBEDDING_ENABLED:-true}" != "true" ]]; then
    log "Skipping embedding setup because EMBEDDING_ENABLED is not true."
    return
  fi

  if python_has_module sentence_transformers; then
    log "sentence-transformers already installed."
  else
    log "Installing embedding runtime dependencies via ${PIP_INDEX_URL}."
    pip_install_with_pypi_fallback sentence-transformers huggingface_hub
  fi

  local embedding_cache_dir="${ROOT_DIR}/${EMBEDDING_MODEL_CACHE_DIR}"
  mkdir -p "${embedding_cache_dir}"
  log "Embedding model cache directory is ready: ${embedding_cache_dir}"
  log "Embedding model will be downloaded on first use if not already cached: ${EMBEDDING_MODEL}"
}

ensure_llm_correction_runtime() {
  if [[ "${LLM_CORRECTION_ENABLED:-false}" != "true" ]]; then
    log "Skipping local LLM correction setup because LLM_CORRECTION_ENABLED is not true."
    return
  fi

  if ! python_has_module llama_cpp; then
    log "Installing llama-cpp-python via ${PIP_INDEX_URL}."
    pip_install llama-cpp-python
  fi

  local model_dir="${ROOT_DIR}/${AUDIO_MODEL_CACHE_ROOT}/llm-correction/${LLM_CORRECTION_MODEL_REPO//\//__}"
  mkdir -p "${model_dir}"
  IFS=',' read -ra model_files <<< "${LLM_CORRECTION_MODEL_FILE}"
  for model_file in "${model_files[@]}"; do
    model_file="${model_file#"${model_file%%[![:space:]]*}"}"
    model_file="${model_file%"${model_file##*[![:space:]]}"}"
    local model_path="${model_dir}/${model_file}"
    if [[ -f "${model_path}" ]]; then
      log "Local LLM correction model file already exists: ${model_path}"
      continue
    fi

    log "Downloading local LLM correction model file: ${LLM_CORRECTION_MODEL_REPO}/${model_file}"
    local model_url="https://huggingface.co/${LLM_CORRECTION_MODEL_REPO}/resolve/main/${model_file}"
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/download_hf_file.py" "${model_url}" "${model_path}"
  done
}

ensure_rag_answer_runtime() {
  if [[ "${RAG_ANSWER_ENABLED:-true}" != "true" ]]; then
    log "Skipping RAG answer setup because RAG_ANSWER_ENABLED is not true."
    return
  fi

  if [[ "${RAG_ANSWER_PROVIDER:-local_llm}" != "local_llm" ]]; then
    log "Skipping local RAG answer setup because RAG_ANSWER_PROVIDER is ${RAG_ANSWER_PROVIDER}."
    return
  fi

  if ! python_has_module llama_cpp; then
    log "Installing llama-cpp-python for local RAG answers via ${PIP_INDEX_URL}."
    pip_install llama-cpp-python
  else
    log "llama-cpp-python already installed."
  fi

  local model_dir="${ROOT_DIR}/${AUDIO_MODEL_CACHE_ROOT}/rag-answer/${RAG_ANSWER_MODEL_REPO//\//__}"
  mkdir -p "${model_dir}"
  IFS=',' read -ra model_files <<< "${RAG_ANSWER_MODEL_FILE}"
  for model_file in "${model_files[@]}"; do
    model_file="${model_file#"${model_file%%[![:space:]]*}"}"
    model_file="${model_file%"${model_file##*[![:space:]]}"}"
    local model_path="${model_dir}/${model_file}"
    if [[ -f "${model_path}" ]]; then
      log "Local RAG answer model file already exists: ${model_path}"
      continue
    fi

    log "Downloading local RAG answer model file: ${RAG_ANSWER_MODEL_REPO}/${model_file}"
    local model_url="https://huggingface.co/${RAG_ANSWER_MODEL_REPO}/resolve/main/${model_file}"
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/download_hf_file.py" "${model_url}" "${model_path}"
  done
}

print_summary() {
  log "Audio dependency environment is ready."
  log "Virtual environment: ${VENV_DIR}"
  log "Python binary: ${VENV_PYTHON}"
  log "To activate manually: source ${VENV_DIR}/bin/activate"
}

main() {
  load_env_file
  ensure_python
  ensure_build_tools
  ensure_postgresql
  ensure_pgvector
  ensure_venv
  if [[ "${VENV_CREATED}" == "1" ]]; then
    upgrade_base_tools
  else
    log "Skipping pip, setuptools, and wheel upgrade for existing virtual environment."
  fi
  ensure_ffmpeg
  ensure_torch
  ensure_numba_runtime
  ensure_whisper
  ensure_funasr
  ensure_transformers_asr
  ensure_qwen_asr
  ensure_pyannote
  ensure_speechbrain
  ensure_pycorrector
  ensure_embedding_runtime
  ensure_llm_correction_runtime
  ensure_rag_answer_runtime
  print_summary
}

main "$@"

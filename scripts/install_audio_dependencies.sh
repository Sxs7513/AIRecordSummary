#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-audio"
PYTHON_BIN_DEFAULT="python3"

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

detect_platform() {
  case "$(uname -s)" in
    Darwin) echo "macos" ;;
    Linux) echo "linux" ;;
    *) echo "unknown" ;;
  esac
}

ensure_python() {
  if command_exists "${PYTHON_BIN_DEFAULT}"; then
    PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
    log "Found Python: ${PYTHON_BIN}"
    return
  fi

  local platform
  platform="$(detect_platform)"

  if [[ "${platform}" == "macos" ]] && command_exists brew; then
    log "Python 3 not found. Installing via Homebrew."
    brew install python
    PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
    return
  fi

  if [[ "${platform}" == "linux" ]]; then
    if command_exists apt-get; then
      log "Python 3 not found. Installing via apt-get."
      sudo apt-get update
      sudo apt-get install -y python3 python3-venv python3-pip
      PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
      return
    fi

    if command_exists dnf; then
      log "Python 3 not found. Installing via dnf."
      sudo dnf install -y python3 python3-pip
      PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
      return
    fi

    if command_exists yum; then
      log "Python 3 not found. Installing via yum."
      sudo yum install -y python3 python3-pip
      PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
      return
    fi
  fi

  fail "Python 3 is required but no supported installer was found."
}

ensure_ffmpeg() {
  if command_exists ffmpeg; then
    log "ffmpeg already installed."
    return
  fi

  local platform
  platform="$(detect_platform)"

  if [[ "${platform}" == "macos" ]] && command_exists brew; then
    log "Installing ffmpeg via Homebrew."
    brew install ffmpeg
    return
  fi

  if [[ "${platform}" == "linux" ]]; then
    if command_exists apt-get; then
      log "Installing ffmpeg via apt-get."
      sudo apt-get update
      sudo apt-get install -y ffmpeg
      return
    fi

    if command_exists dnf; then
      log "Installing ffmpeg via dnf."
      sudo dnf install -y ffmpeg
      return
    fi

    if command_exists yum; then
      log "Installing ffmpeg via yum."
      sudo yum install -y ffmpeg
      return
    fi
  fi

  fail "ffmpeg is required but no supported installer was found."
}

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtual environment at ${VENV_DIR}."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  else
    log "Virtual environment already exists at ${VENV_DIR}."
  fi

  VENV_PYTHON="${VENV_DIR}/bin/python"
  VENV_PIP="${VENV_DIR}/bin/pip"

  if [[ ! -x "${VENV_PYTHON}" ]]; then
    fail "Virtual environment python binary not found at ${VENV_PYTHON}."
  fi
}

upgrade_base_tools() {
  log "Upgrading pip, setuptools, and wheel in virtual environment."
  "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
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

  log "Installing torch."
  "${VENV_PIP}" install torch
}

ensure_whisper() {
  if python_has_module whisper; then
    log "whisper already installed."
    return
  fi

  log "Installing openai-whisper from official GitHub repository."
  "${VENV_PIP}" install "git+https://github.com/openai/whisper.git"
}

ensure_pyannote() {
  if python_has_module pyannote.audio; then
    log "pyannote.audio already installed."
    return
  fi

  log "Installing pyannote.audio."
  "${VENV_PIP}" install pyannote.audio
}

ensure_speechbrain() {
  if python_has_module speechbrain; then
    log "speechbrain already installed."
    return
  fi

  log "Installing SpeechBrain."
  "${VENV_PIP}" install speechbrain
}

print_summary() {
  log "Audio dependency environment is ready."
  log "Virtual environment: ${VENV_DIR}"
  log "Python binary: ${VENV_PYTHON}"
  log "To activate manually: source ${VENV_DIR}/bin/activate"
}

main() {
  ensure_python
  ensure_ffmpeg
  ensure_venv
  upgrade_base_tools
  ensure_torch
  ensure_whisper
  ensure_pyannote
  ensure_speechbrain
  print_summary
}

main "$@"

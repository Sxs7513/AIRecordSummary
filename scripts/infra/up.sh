#!/usr/bin/env bash

# Start the local container runtime (Colima when available) before Compose.
# This keeps `npm run infra:up` usable immediately after a macOS restart.
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_docker_ready() {
  docker info >/dev/null 2>&1
}

if ! is_docker_ready; then
  if ! command -v colima >/dev/null 2>&1; then
    echo "Docker is not running, and Colima is not installed or not on PATH." >&2
    echo "Start Docker Desktop, or install Colima and run: colima start" >&2
    exit 1
  fi

  echo "Docker daemon is unavailable; starting Colima..."
  colima start

  for _ in $(seq 1 60); do
    if is_docker_ready; then
      break
    fi
    sleep 1
  done

  if ! is_docker_ready; then
    echo "Docker did not become ready within 60 seconds after starting Colima." >&2
    exit 1
  fi
fi

bash "$script_dir/compose.sh" up -d --wait

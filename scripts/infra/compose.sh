#!/usr/bin/env bash

set -Eeuo pipefail

if docker compose version >/dev/null 2>&1; then
  exec docker compose "$@"
elif command -v docker-compose >/dev/null 2>&1; then
  exec docker-compose "$@"
else
  echo "Docker Compose is required. Install the Docker Compose plugin or docker-compose." >&2
  exit 1
fi

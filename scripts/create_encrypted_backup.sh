#!/usr/bin/env bash

set -Eeuo pipefail

umask 077

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BACKUP_DIR="${PROJECT_ROOT}/encrypted-backups"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

require_command() {
  local command_name="$1"
  local install_hint="${2:-}"

  if ! command -v "${command_name}" >/dev/null 2>&1; then
    if [[ -n "${install_hint}" ]]; then
      fail "缺少命令 ${command_name}。${install_hint}"
    fi
    fail "缺少命令 ${command_name}。"
  fi
}

load_database_config() {
  [[ -f "${PROJECT_ROOT}/.env" ]] || fail "找不到 ${PROJECT_ROOT}/.env"

  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  if [[ -f "${PROJECT_ROOT}/.env.local" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env.local"
  fi

  local variable_name
  for variable_name in DB_HOST DB_PORT DB_USER DB_PASSWORD DB_NAME; do
    [[ -n "${!variable_name:-}" ]] || fail ".env 中缺少 ${variable_name}"
  done
}

cleanup() {
  if [[ -n "${WORK_DIR:-}" && -d "${WORK_DIR}" ]]; then
    rm -rf -- "${WORK_DIR}"
  fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
用法：./scripts/create_encrypted_backup.sh

将 PostgreSQL 数据库导出、压缩并使用密码加密，
生成 encrypted-backups/*.tar.zst.age。密码由 age 在终端中交互式读取。
录音文件不会被备份；恢复数据库后，重新上传相同 MD5 的录音会补回源文件且不重跑处理流程。
EOF
  exit 0
fi

[[ $# -eq 0 ]] || fail "不支持参数；使用 --help 查看用法。"

require_command pg_dump "请先安装 PostgreSQL 客户端。"
require_command tar
require_command zstd "macOS 可运行：brew install zstd"
require_command age "macOS 可运行：brew install age"
require_command shasum

load_database_config

mkdir -p "${BACKUP_DIR}"
chmod 700 "${BACKUP_DIR}"

readonly TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
readonly BACKUP_NAME="ai-record-summary-${TIMESTAMP}.tar.zst.age"
readonly FINAL_BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"
[[ ! -e "${FINAL_BACKUP_PATH}" ]] || fail "备份文件已存在：${FINAL_BACKUP_PATH}"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ai-record-summary-backup.XXXXXX")"
readonly WORK_DIR
trap cleanup EXIT INT TERM

readonly DATABASE_DUMP_PATH="${WORK_DIR}/database.dump"
readonly MANIFEST_PATH="${WORK_DIR}/manifest.txt"
readonly ENCRYPTED_TEMP_PATH="${WORK_DIR}/${BACKUP_NAME}"

case "${DB_SSL:-false}" in
  1|true|TRUE|yes|YES|on|ON) readonly PG_SSL_MODE="require" ;;
  *) readonly PG_SSL_MODE="prefer" ;;
esac

log "[1/3] 正在导出数据库 ${DB_NAME}（${DB_HOST}:${DB_PORT}）..."
PGPASSWORD="${DB_PASSWORD}" PGSSLMODE="${PG_SSL_MODE}" pg_dump \
  --host="${DB_HOST}" \
  --port="${DB_PORT}" \
  --username="${DB_USER}" \
  --dbname="${DB_NAME}" \
  --no-password \
  --format=custom \
  --compress=0 \
  --file="${DATABASE_DUMP_PATH}"

readonly DATABASE_BYTES="$(wc -c < "${DATABASE_DUMP_PATH}" | tr -d '[:space:]')"

{
  printf 'format_version=2\n'
  printf 'created_at_utc=%s\n' "${TIMESTAMP}"
  printf 'database_name=%s\n' "${DB_NAME}"
  printf 'database_dump_format=postgresql_custom\n'
  printf 'database_dump_bytes=%s\n' "${DATABASE_BYTES}"
  printf 'backup_scope=database_only\n'
  printf 'source_audio_restore=duplicate_upload_self_heal\n'
  printf 'pg_dump_version=%s\n' "$(pg_dump --version)"
} > "${MANIFEST_PATH}"

log "[2/3] 正在打包并加密数据库导出..."
log "age 接下来会要求输入并确认备份密码；请妥善保存，密码丢失后无法恢复。"
tar -cf - \
  -C "${WORK_DIR}" database.dump manifest.txt \
  | zstd --threads=0 --quiet -10 \
  | age --passphrase -o "${ENCRYPTED_TEMP_PATH}"

[[ -s "${ENCRYPTED_TEMP_PATH}" ]] || fail "加密文件没有生成或内容为空。"
mv -- "${ENCRYPTED_TEMP_PATH}" "${FINAL_BACKUP_PATH}"
chmod 600 "${FINAL_BACKUP_PATH}"

readonly BACKUP_BYTES="$(wc -c < "${FINAL_BACKUP_PATH}" | tr -d '[:space:]')"
readonly BACKUP_SHA256="$(shasum -a 256 "${FINAL_BACKUP_PATH}" | awk '{print $1}')"

log "[3/3] 备份完成。"
printf '文件：%s\n' "${FINAL_BACKUP_PATH}"
printf '大小：%s bytes\n' "${BACKUP_BYTES}"
printf 'SHA-256：%s\n' "${BACKUP_SHA256}"

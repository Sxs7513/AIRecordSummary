#!/usr/bin/env bash

set -Eeuo pipefail

umask 077

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

fail() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "缺少命令 $1。"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
用法：./scripts/decrypt_encrypted_backup.sh [--restore-database] [--yes] <备份文件.tar.zst.age> [保留解包结果的目录]

使用备份密码解密并解包数据库导出文件。

--restore-database 会用 .env / .env.local 中的 DB_* 配置恢复 database.dump。
该操作会清空并重建目标数据库中的对象；交互式执行时需要确认，
自动化执行时须额外传入 --yes。

恢复模式默认仅在系统临时目录解密，完成后自动清理，不会留下未加密副本。
若不执行恢复，必须显式传入解包目录；该目录不会被覆盖。
录音文件不包含在新备份中，且恢复时不会触碰 uploads/recordings。恢复后可重新上传相同
MD5 的录音补回源文件，不会重跑已有处理结果。
EOF
  exit 0
fi

RESTORE_DATABASE=false
ASSUME_YES=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --restore-database) RESTORE_DATABASE=true; shift ;;
    --yes) ASSUME_YES=true; shift ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        POSITIONAL+=("$1")
        shift
      done
      ;;
    -*) fail "不支持的参数：$1" ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

[[ ${#POSITIONAL[@]} -ge 1 && ${#POSITIONAL[@]} -le 2 ]] || fail "用法：$0 [--restore-database] [--yes] <备份文件.tar.zst.age> [解包目录]"
[[ "${RESTORE_DATABASE}" == true || ${#POSITIONAL[@]} -eq 2 ]] || fail "仅解密查看时必须显式指定解包目录，避免在项目中留下未加密副本。"

readonly BACKUP_PATH="${POSITIONAL[0]}"
[[ -f "${BACKUP_PATH}" ]] || fail "找不到备份文件：${BACKUP_PATH}"
[[ "${BACKUP_PATH}" == *.tar.zst.age ]] || fail "备份文件必须以 .tar.zst.age 结尾。"

require_command age
require_command zstd
require_command tar
if [[ "${RESTORE_DATABASE}" == true ]]; then
  require_command pg_restore
fi

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

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ai-record-summary-restore.XXXXXX")"
cleanup() {
  [[ -d "${WORK_DIR}" ]] && rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT INT TERM

if [[ ${#POSITIONAL[@]} -eq 2 ]]; then
  OUTPUT_DIR="${POSITIONAL[1]}"
  [[ ! -e "${OUTPUT_DIR}" ]] || fail "解包目录已存在，为避免覆盖已停止：${OUTPUT_DIR}"
  KEEP_DECRYPTED_OUTPUT=true
else
  OUTPUT_DIR="${WORK_DIR}/output"
  KEEP_DECRYPTED_OUTPUT=false
fi
readonly OUTPUT_DIR KEEP_DECRYPTED_OUTPUT

readonly COMPRESSED_PATH="${WORK_DIR}/backup.tar.zst"
readonly TAR_PATH="${WORK_DIR}/backup.tar"
readonly EXTRACT_DIR="${WORK_DIR}/unpacked"

printf '正在解密备份，age 会要求输入备份密码…\n'
age --decrypt -o "${COMPRESSED_PATH}" "${BACKUP_PATH}"
zstd --decompress --quiet -o "${TAR_PATH}" "${COMPRESSED_PATH}"

while IFS= read -r archive_path; do
  [[ "${archive_path}" == "database.dump" || "${archive_path}" == "manifest.txt" || "${archive_path}" == uploads/recordings/* ]] \
    || fail "备份归档包含不允许的路径：${archive_path}"
done < <(tar -tf "${TAR_PATH}")

mkdir -p "${EXTRACT_DIR}"
tar -xf "${TAR_PATH}" -C "${EXTRACT_DIR}"
[[ -f "${EXTRACT_DIR}/database.dump" ]] || fail "备份中缺少 database.dump"
[[ -f "${EXTRACT_DIR}/manifest.txt" ]] || fail "备份中缺少 manifest.txt"

mkdir -p "$(dirname "${OUTPUT_DIR}")"
mv "${EXTRACT_DIR}" "${OUTPUT_DIR}"
if [[ "${KEEP_DECRYPTED_OUTPUT}" == true ]]; then
  printf '解密并解包完成：%s\n' "${OUTPUT_DIR}"
  printf '数据库导出：%s/database.dump\n' "${OUTPUT_DIR}"
fi

if [[ "${RESTORE_DATABASE}" == true ]]; then
  load_database_config
  case "${DB_SSL:-false}" in
    1|true|TRUE|yes|YES|on|ON) PG_SSL_MODE="require" ;;
    *) PG_SSL_MODE="prefer" ;;
  esac

  if [[ "${ASSUME_YES}" != true ]]; then
    printf '\n即将清空并恢复 PostgreSQL 数据库 %s（%s:%s）；不会触碰 uploads/recordings。\n' "${DB_NAME}" "${DB_HOST}" "${DB_PORT}" >&2
    read -r -p '输入数据库名以确认：' confirmation
    [[ "${confirmation}" == "${DB_NAME}" ]] || fail "确认内容不匹配，已取消数据库恢复。"
  fi

  printf '正在恢复数据库 %s…\n' "${DB_NAME}"
  PGPASSWORD="${DB_PASSWORD}" PGSSLMODE="${PG_SSL_MODE}" pg_restore \
    --host="${DB_HOST}" \
    --port="${DB_PORT}" \
    --username="${DB_USER}" \
    --dbname="${DB_NAME}" \
    --no-password \
    --clean \
    --if-exists \
    --no-owner \
    --no-privileges \
    "${OUTPUT_DIR}/database.dump"
  printf '数据库恢复完成：%s\n' "${DB_NAME}"
  printf '录音未从备份恢复；需要时重新上传相同 MD5 的录音即可补回源文件。\n'
fi

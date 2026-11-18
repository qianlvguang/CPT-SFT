#!/usr/bin/env bash
# Shared shell core utilities for CPT-SFT B300 launch scripts.
# Intended to be sourced by hardware profiles and launchers.

if [[ -n ${_CPT_SFT_CORE_SH_LOADED:-} ]]; then
  return 0 2>/dev/null || exit 0
fi
_CPT_SFT_CORE_SH_LOADED=1

log_info() {
  printf '[INFO] %s\n' "$*" >&2
}

log_warn() {
  printf '[WARN] %s\n' "$*" >&2
}

log_err() {
  printf '[ERROR] %s\n' "$*" >&2
}

die() {
  local code=1
  if [[ $# -gt 1 && $1 =~ ^[0-9]+$ ]]; then
    code=$1
    shift
  fi
  log_err "$*"
  exit "$code"
}

resolve_repo_root() {
  local start_dir="${1:-$(pwd)}"
  local dir
  dir="$(cd "$start_dir" && pwd)"
  while [[ "$dir" != "/" ]]; do
    if [[ -d "$dir/training" && -d "$dir/docs" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

format_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+="$(printf '%q ' "$arg")"
  done
  printf '%s' "${out% }"
}

print_or_exec_cmd() {
  local dry_run="${DRY_RUN:-1}"
  local launch="${LAUNCH:-0}"

  # Explicit DRY_RUN=1 always wins; otherwise require LAUNCH=1 or DRY_RUN=0 to execute.
  if [[ "$dry_run" == "1" ]] || [[ "$launch" != "1" && "$dry_run" != "0" ]]; then
    printf '%s\n' "$(format_cmd "$@")"
    return 0
  fi

  local bin="$1"
  if ! command -v "$bin" >/dev/null 2>&1; then
    die 1 "Cannot find executable '$bin'. Activate the target training environment first."
  fi
  exec "$@"
}

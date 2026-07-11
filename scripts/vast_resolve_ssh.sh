#!/usr/bin/env bash
# Resolve Vast.ai SSH host/port via CLI (proxy or direct — may differ from DeployGPU dashboard).
#
# Usage:
#   source scripts/vast_resolve_ssh.sh && vast_resolve_ssh 44470343
#   echo "$VAST_SSH_HOST $VAST_SSH_PORT"
#
# Sets: VAST_SSH_HOST, VAST_SSH_PORT, VAST_SSH_USER, VAST_SSH_CMD
# Returns 0 on success, 1 if vastai unavailable or instance not reachable via CLI.
set -euo pipefail

vast_resolve_ssh() {
  local instance_id="${1:-${VAST_INSTANCE_ID:-}}"
  if [[ -z "$instance_id" ]]; then
    echo "vast_resolve_ssh: no instance id" >&2
    return 1
  fi

  if ! command -v vastai &>/dev/null; then
    echo "vast_resolve_ssh: vastai CLI not installed (pip install vastai)" >&2
    return 1
  fi

  local api_key_file="${VAST_API_KEY_FILE:-$HOME/.config/vastai/vast_api_key}"
  if [[ ! -f "$api_key_file" ]]; then
    echo "vast_resolve_ssh: no API key at $api_key_file" >&2
    echo "  Create one: Vast.ai → Account → CLI → copy key to $api_key_file" >&2
    return 1
  fi

  local raw_json host port user
  if raw_json=$(vastai ssh-url "$instance_id" --raw 2>/dev/null); then
    if command -v python3 &>/dev/null; then
      host=$(printf '%s' "$raw_json" | python3 -c "
import json,sys
d=json.load(sys.stdin)
# ssh-url raw may be a string command or dict with host/port
if isinstance(d, str):
    print('CMD', d)
elif isinstance(d, dict):
    print(d.get('host',''), d.get('port',''), d.get('user','root'))
" 2>/dev/null || true)
      if [[ "$host" == CMD* ]]; then
        VAST_SSH_CMD="${host#CMD }"
      elif [[ -n "$host" ]]; then
        read -r VAST_SSH_HOST VAST_SSH_PORT VAST_SSH_USER <<< "$host"
      fi
    fi
  fi

  if [[ -z "${VAST_SSH_CMD:-}" && -z "${VAST_SSH_HOST:-}" ]]; then
    local url_out
    url_out=$(vastai ssh-url "$instance_id" 2>/dev/null) || {
      echo "vast_resolve_ssh: vastai ssh-url $instance_id failed (login? instance running?)" >&2
      return 1
    }
    # Typical: "ssh -p 12345 root@1.2.3.4" or "ssh root@host -p port"
    if [[ "$url_out" =~ -p[[:space:]]+([0-9]+)[[:space:]]+([^@]+)@([^[:space:]]+) ]]; then
      VAST_SSH_PORT="${BASH_REMATCH[1]}"
      VAST_SSH_USER="${BASH_REMATCH[2]}"
      VAST_SSH_HOST="${BASH_REMATCH[3]}"
    elif [[ "$url_out" =~ ([^@]+)@([^[:space:]]+)[[:space:]]+-p[[:space:]]+([0-9]+) ]]; then
      VAST_SSH_USER="${BASH_REMATCH[1]}"
      VAST_SSH_HOST="${BASH_REMATCH[2]}"
      VAST_SSH_PORT="${BASH_REMATCH[3]}"
    elif [[ "$url_out" =~ ssh[[:space:]]+(.*) ]]; then
      VAST_SSH_CMD="${BASH_REMATCH[1]}"
    else
      echo "vast_resolve_ssh: could not parse: $url_out" >&2
      return 1
    fi
  fi

  VAST_SSH_USER="${VAST_SSH_USER:-root}"
  if [[ -n "${VAST_SSH_CMD:-}" ]]; then
    export VAST_SSH_CMD
    echo "Vast SSH: $VAST_SSH_CMD"
  else
    export VAST_SSH_HOST VAST_SSH_PORT VAST_SSH_USER
    VAST_SSH_CMD="ssh -p ${VAST_SSH_PORT} ${VAST_SSH_USER}@${VAST_SSH_HOST}"
    export VAST_SSH_CMD
    echo "Vast SSH: $VAST_SSH_CMD"
  fi
  return 0
}

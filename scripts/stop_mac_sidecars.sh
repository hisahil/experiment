#!/usr/bin/env bash
# Stop Mac sidecars: live pod sync + hourly snapshots.
set -euo pipefail

UID_NUM="$(id -u)"
GUI="gui/${UID_NUM}"

for label in com.msc_thesis.sync_pod_live com.msc_thesis.hourly_pod_snapshot; do
  launchctl bootout "${GUI}/${label}" 2>/dev/null || true
done

pkill -f 'sync_pod_live.sh' 2>/dev/null || true
pkill -f 'hourly_pod_snapshot.sh' 2>/dev/null || true

echo "Stopped backup sidecars."
launchctl list 2>/dev/null | grep msc_thesis || echo "No msc_thesis launchd jobs."
pgrep -fl 'sync_pod_live|hourly_pod_snapshot' 2>/dev/null || echo "No backup processes."

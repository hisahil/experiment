#!/usr/bin/env bash
# Pull 1.5B tagfix artifacts from RunPod to Mac (chunked base64 tar, SHA verified).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POD="${POD:-020gpg3omasfwu-64410d4f@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/deploygpu_ed25519}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$ROOT/backup_gpu_live/tagfix_1.5b_pull_${STAMP}"
mkdir -p "$DEST"

echo "==> Prepare tarball + b64 chunks on pod..."
ssh -tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=120 "$POD" <<'REMOTE' 2>/dev/null | tr -d '\r' | grep -E '^(SHA|BYTES|CHUNKS)=' 
cd /root/msc_thesis_experiment
tar czf /tmp/tagfix_1.5b_pull.tar.gz \
  data/1.5b data/baselines/1.5b results/1.5b data/logs/full_1.5b_tagfix_rerun.log 2>/dev/null
rm -f /tmp/b64chunk.*
base64 -w0 /tmp/tagfix_1.5b_pull.tar.gz | split -b 400000 - /tmp/b64chunk.
echo SHA=$(sha256sum /tmp/tagfix_1.5b_pull.tar.gz | awk '{print $1}')
echo BYTES=$(stat -c %s /tmp/tagfix_1.5b_pull.tar.gz)
echo CHUNKS=$(ls -1 /tmp/b64chunk.* 2>/dev/null | wc -l | tr -d ' ')
exit 0
REMOTE

POD_SHA=$(ssh -tt -i "$KEY" -o StrictHostKeyChecking=no "$POD" "sha256sum /tmp/tagfix_1.5b_pull.tar.gz | awk '{print \$1}'; exit" 2>/dev/null | tr -d '\r' | grep -E '^[a-f0-9]{64}$' | tail -1)
POD_BYTES=$(ssh -tt -i "$KEY" -o StrictHostKeyChecking=no "$POD" "stat -c %s /tmp/tagfix_1.5b_pull.tar.gz; exit" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | tail -1)
CHUNK_LIST=$(ssh -tt -i "$KEY" -o StrictHostKeyChecking=no "$POD" "ls -1 /tmp/b64chunk.*; exit" 2>/dev/null | tr -d '\r' | grep '/tmp/b64chunk\.' | sort)

echo "Pod tarball: ${POD_BYTES} bytes sha256=${POD_SHA}"
echo "$POD_SHA $POD_BYTES" > "$DEST/expected_sha256.txt"

B64_OUT="$DEST/pull.b64"
: > "$B64_OUT"
i=0
while IFS= read -r chunk_path; do
  [[ -z "$chunk_path" ]] && continue
  i=$((i + 1))
  echo "==> Download chunk $i: $chunk_path"
  ssh -tt -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=120 "$POD" "cat ${chunk_path}; exit" 2>/dev/null \
    | tr -d '\r' \
    | sed $'s/\033\\[[0-9;?]*[a-zA-Z]//g' \
    | grep -E '^[A-Za-z0-9+/=]+$' >> "$B64_OUT"
done <<< "$CHUNK_LIST"

echo "==> Decode b64 (${POD_BYTES} bytes expected)..."
base64 -D -i "$B64_OUT" -o "$DEST/tagfix_1.5b_pull.tar.gz" 2>/dev/null || base64 -d -i "$B64_OUT" -o "$DEST/tagfix_1.5b_pull.tar.gz"
LOCAL_SHA=$(shasum -a 256 "$DEST/tagfix_1.5b_pull.tar.gz" | awk '{print $1}')
LOCAL_BYTES=$(stat -f%z "$DEST/tagfix_1.5b_pull.tar.gz" 2>/dev/null || stat -c%s "$DEST/tagfix_1.5b_pull.tar.gz")
echo "Local: ${LOCAL_BYTES} bytes sha256=${LOCAL_SHA}"
if [[ "$LOCAL_SHA" != "$POD_SHA" ]]; then
  echo "SHA FAIL expected $POD_SHA got $LOCAL_SHA" >&2
  exit 1
fi
echo "SHA OK"
tar xzf "$DEST/tagfix_1.5b_pull.tar.gz" -C "$DEST"
echo "$LOCAL_SHA  $LOCAL_BYTES bytes" > "$DEST/PULL_SHA256.txt"

echo "==> Syncing to workspace..."
for f in complete_dialogues.jsonl post_injection_runs.jsonl post_injection_runs.rejudged.jsonl; do
  src="$DEST/data/1.5b/$f"
  dst="$ROOT/data/1.5b/$f"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dst"
    echo "  synced $f ($(wc -l < "$dst" | tr -d ' ') lines)"
  fi
done
mkdir -p "$ROOT/results/1.5b" "$ROOT/backup_gpu_live/1.5b/data/1.5b" "$ROOT/backup_gpu_live/1.5b/results/1.5b"
cp -f "$DEST/data/1.5b/"*.jsonl "$ROOT/backup_gpu_live/1.5b/data/1.5b/" 2>/dev/null || true
cp -rf "$DEST/results/1.5b/"* "$ROOT/results/1.5b/" 2>/dev/null || true
cp -rf "$DEST/results/1.5b/"* "$ROOT/backup_gpu_live/1.5b/results/1.5b/" 2>/dev/null || true
mkdir -p "$ROOT/data/logs"
cp -f "$DEST/data/logs/full_1.5b_tagfix_rerun.log" "$ROOT/data/logs/" 2>/dev/null || true

echo "==> Verify Mac"
wc -l "$ROOT/data/1.5b/complete_dialogues.jsonl" "$ROOT/data/1.5b/post_injection_runs.jsonl" "$ROOT/data/1.5b/post_injection_runs.rejudged.jsonl"
python3 -c "
import json
fb=sum(1 for l in open('$ROOT/data/1.5b/post_injection_runs.rejudged.jsonl') if l.strip() and json.loads(l).get('judge_fallback'))
print('fallbacks', fb)
"
echo "==> DONE $DEST"

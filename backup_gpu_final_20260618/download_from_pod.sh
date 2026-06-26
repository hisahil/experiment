#!/usr/bin/env bash
set -euo pipefail
KEY=~/.ssh/deploygpu_ed25519
HOST=020gpg3omasfwu-64410d4f@ssh.runpod.io
ROOT="/Users/sararoy/msc_thesis_experiment/backup_gpu_final_20260618"
CHUNKS="$ROOT/chunks"
mkdir -p "$CHUNKS"

download_chunk() {
  local prefix="$1"   # bkp18 or bkp19
  local suf="$2"
  local remote="/tmp/${prefix}_${suf}"
  local localf="$CHUNKS/${prefix}_${suf}"
  local sz=0
  if [[ -f "$localf" ]]; then
    sz=$(wc -c < "$localf" | tr -d ' ')
    if [[ "$sz" -ge 1000 ]]; then
      echo "skip ${prefix}_${suf} ($sz bytes)"
      return 0
    fi
  fi
  echo "downloading ${prefix}_${suf}..."
  ssh -tt -o ConnectTimeout=60 -i "$KEY" "$HOST" <<EOF 2>/dev/null | tr -d '\r' | perl -pe 's/\e\[[\d;?]*[a-zA-Z]//g' | grep -oE '[A-Za-z0-9+/=]{100,}' | head -1 | base64 -d > "$localf"
base64 -w 0 ${remote}
exit
EOF
  sz=$(wc -c < "$localf" | tr -d ' ')
  echo "  got $sz bytes"
}

# bkp18: main data tar (19 chunks aa-as)
for s in aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar as; do
  download_chunk bkp18 "$s"
done

# bkp19: gpu_run_script34.log (5 chunks)
for s in aa ab ac ad ae; do
  download_chunk bkp19 "$s"
done

# gpu_run.log (small, single base64)
echo "downloading gpu_run.log..."
ssh -tt -o ConnectTimeout=60 -i "$KEY" "$HOST" <<'EOF' 2>/dev/null | tr -d '\r' | perl -pe 's/\e\[[\d;?]*[a-zA-Z]//g' | grep -oE '[A-Za-z0-9+/=]{100,}' | head -1 | base64 -d > "$ROOT/root/gpu_run.log"
base64 -w 0 /tmp/gpu_run.log
exit
EOF
echo "  gpu_run.log: $(wc -c < "$ROOT/root/gpu_run.log" | tr -d ' ') bytes"

# Assemble tar
TAR="$ROOT/backup_final_data.tar.gz"
cat "$CHUNKS"/bkp18_* > "$TAR"
echo "TAR assembled: $(wc -c < "$TAR" | tr -d ' ') bytes (expect 9473418)"

# Extract
tar xzf "$TAR" -C "$ROOT"
echo "Extracted data/ and results/"

# Assemble log
cat "$CHUNKS"/bkp19_* > "$ROOT/root/gpu_run_script34.log"
echo "gpu_run_script34.log: $(wc -c < "$ROOT/root/gpu_run_script34.log" | tr -d ' ') bytes (expect 2211164)"

echo "DOWNLOAD_COMPLETE"

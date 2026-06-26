#!/bin/bash
set -euo pipefail
KEY=~/.ssh/deploygpu_ed25519
HOST=020gpg3omasfwu-64410d4f@ssh.runpod.io
OUTDIR="/Users/sararoy/msc_thesis_experiment/backup_gpu_20260617/chunks"
TAR="/Users/sararoy/msc_thesis_experiment/backup_gpu_20260617/backup_script12.tar.gz"
mkdir -p "$OUTDIR"

suffixes=(aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar as at au av aw ax ay az ba bb bc bd be bf bg)

download_chunk() {
  local suf="$1"
  local remote="/tmp/bkp16_${suf}"
  local localf="$OUTDIR/bkp16_${suf}"
  local sz=0
  if [[ -f "$localf" ]]; then
    sz=$(wc -c < "$localf" | tr -d ' ')
    if [[ "$sz" -ge 1000 ]]; then
      echo "skip $suf ($sz bytes)"
      return 0
    fi
  fi
  echo "downloading $suf..."
  ssh -tt -o ConnectTimeout=30 -i "$KEY" "$HOST" <<EOF 2>/dev/null | tr -d '\r' | perl -pe 's/\e\[[\d;?]*[a-zA-Z]//g' | grep -oE '[A-Za-z0-9+/=]{100,}' | head -1 | base64 -d > "$localf"
base64 -w 0 ${remote}
exit
EOF
  sz=$(wc -c < "$localf" | tr -d ' ')
  echo "  got $sz bytes"
}

for s in "${suffixes[@]}"; do
  download_chunk "$s"
done

cat "$OUTDIR"/bkp16_* > "$TAR"
echo "TAR size: $(wc -c < "$TAR" | tr -d ' ')"

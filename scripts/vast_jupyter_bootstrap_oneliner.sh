#!/usr/bin/env bash
# Print exact Vast.ai Jupyter Terminal bootstrap steps when direct SSH is blocked.
#
# Run on Mac from repo root:
#   bash scripts/vast_jupyter_bootstrap_oneliner.sh
#   bash scripts/vast_jupyter_bootstrap_oneliner.sh --bundle   # also write base64 upload bundle
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/data/deploygpu_instance.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE"; set +a
fi

INSTANCE_ID="${VAST_INSTANCE_ID:-44470343}"
REPO_URL="${OLMO_REPO_URL:-https://github.com/sahil-ejaz/experiment.git}"

MAKE_BUNDLE=0
[[ "${1:-}" == "--bundle" ]] && MAKE_BUNDLE=1

cat <<EOF

================================================================================
Vast.ai OLMo bootstrap (Instance ${INSTANCE_ID}) — Jupyter Terminal workaround
================================================================================

WHY direct SSH fails
--------------------
DeployGPU shows Quick Connect: ssh root@${DEPLOYGPU_HOST:-212.13.234.23} -p ${DEPLOYGPU_PORT:-30342}
From your Mac, that host has NO reachable TCP ports (timeout on 22, 30342, 8080, …).
The instance IS running (Jupyter / Instance Portal work) — only the host's *direct*
port mapping to your IP is broken or firewalled. Vast proxy SSH (vastai ssh-url) or
Jupyter Terminal bypass this.

YES — use Instance Portal / Jupyter Terminal
--------------------------------------------
1. Open https://cloud.vast.ai/instances/
2. Click instance ${INSTANCE_ID} (A100-SXM4-40GB)
3. Click **Jupyter** or **Instance Portal** → open **Terminal** (New → Terminal)
4. Paste ONE of the blocks below into that terminal.

================================================================================
OPTION A — git clone + setup (simplest if GitHub repo is up to date)
================================================================================

EOF

cat <<'PASTE_A'
set -euo pipefail
export OLMO_REPO_DIR="$HOME/msc_thesis_experiment"
if [[ ! -f "$OLMO_REPO_DIR/scripts/olmo_pod_setup.sh" ]]; then
  git clone --depth 1 https://github.com/sahil-ejaz/experiment.git "$OLMO_REPO_DIR"
fi
cd "$OLMO_REPO_DIR"
bash scripts/olmo_pod_setup.sh
PASTE_A

cat <<EOF

================================================================================
OPTION B — fetch setup script only (if olmo_pod_setup.sh is on GitHub main)
================================================================================

EOF

cat <<PASTE_B
set -euo pipefail
mkdir -p ~/msc_thesis_experiment/scripts
curl -fsSL "https://raw.githubusercontent.com/sahil-ejaz/experiment/main/scripts/olmo_pod_setup.sh" \\
  -o ~/msc_thesis_experiment/scripts/olmo_pod_setup.sh
bash ~/msc_thesis_experiment/scripts/olmo_pod_setup.sh
PASTE_B

if [[ "$MAKE_BUNDLE" == "1" ]]; then
  BUNDLE="$ROOT/data/olmo_jupyter_upload.tgz"
  echo ""
  echo "Building minimal upload bundle: $BUNDLE"
  tar czf "$BUNDLE" \
    -C "$ROOT" \
    scripts/olmo_pod_setup.sh \
    scripts/setup_gpu.sh \
    scripts/generate_baselines.py \
    src/ \
    data/smoke_tasks.jsonl \
    configs/model_families.yaml \
    configs/experiment.yaml \
    configs/baseline_generation.yaml \
    prompts/baseline_generation.txt \
    prompts/baseline_generation_gemma.txt

  B64="$ROOT/data/olmo_jupyter_upload.b64"
  base64 < "$BUNDLE" > "$B64"
  echo "Wrote $BUNDLE ($(du -h "$BUNDLE" | cut -f1)) and $B64"
  echo ""
  cat <<EOF
================================================================================
OPTION C — base64 upload (local-only changes not pushed to GitHub)
================================================================================
On Mac the bundle was written to: $B64
Copy its contents, then in Jupyter Terminal:

EOF
  cat <<'PASTE_C'
set -euo pipefail
mkdir -p ~/msc_thesis_experiment
base64 -d > /tmp/olmo_jupyter_upload.tgz <<'B64EOF'
PASTE_C
  echo "(paste contents of data/olmo_jupyter_upload.b64 here, then close with B64EOF on its own line)"
  cat <<'PASTE_C2'
B64EOF
tar xzf /tmp/olmo_jupyter_upload.tgz -C ~/msc_thesis_experiment
cd ~/msc_thesis_experiment
bash scripts/olmo_pod_setup.sh
PASTE_C2
fi

cat <<EOF

================================================================================
After setup (~15-25 min first time)
================================================================================
  tmux attach -t olmo_smoke          # watch 5-task smoke
  tail -f logs/olmo_baseline_smoke_*.log

Fix direct SSH from Mac (optional, for rsync later)
---------------------------------------------------
1. Vast.ai → Account → CLI → create API key
2. echo 'YOUR_KEY' > ~/.config/vastai/vast_api_key && chmod 600 ~/.config/vastai/vast_api_key
3. vastai ssh-url ${INSTANCE_ID}
4. vastai ssh ${INSTANCE_ID}          # interactive proxy SSH (works when host ports don't)
5. bash scripts/bootstrap_deploygpu_olmo.sh   # auto rsync + setup once SSH works

DeployGPU dashboard Quick Connect (${DEPLOYGPU_HOST:-212.13.234.23}:${DEPLOYGPU_PORT:-30342}) may stay
wrong/unreachable even when vastai ssh-url gives a working proxy endpoint.

EOF

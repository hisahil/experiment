#!/usr/bin/env bash
# Print a single curl|bash one-liner for Vast Jupyter Terminal (raw.githubusercontent.com).
# Run on Mac: bash scripts/olmo_remote_one_liner.sh
# Optional: OLMO_GIT_REF=main (default) or commit SHA after push.
set -euo pipefail
echo "# NOTE: sahil-ejaz/experiment is PRIVATE — raw.githubusercontent.com returns 404."
echo "# Use: bash scripts/vast_jupyter_bootstrap_oneliner.sh --bundle  (base64 paste in Jupyter)"
echo ""
REF="${OLMO_GIT_REF:-v2-improvements}"
REPO="${OLMO_REPO_URL:-https://github.com/sahil-ejaz/experiment.git}"
RAW="https://raw.githubusercontent.com/sahil-ejaz/experiment/${REF}/scripts/olmo_pod_setup.sh"
ONELINER="curl -fsSL \"${RAW}\" | bash -s --"
echo "$ONELINER"
echo ""
echo "# Full clone fallback if raw script missing on ${REF}:"
echo "git clone --depth 1 \"${REPO}\" \"\$HOME/msc_thesis_experiment\" && bash \"\$HOME/msc_thesis_experiment/scripts/olmo_pod_setup.sh\""

#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
print_message() { echo -e "${2}${1}${NC}"; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nocodeml

print_message "Starting No‑Code ML Platform…" "$GREEN"
print_message "Opening your default browser – press Ctrl+C to quit." "$GREEN"

streamlit run app.py --server.enableXsrfProtection false

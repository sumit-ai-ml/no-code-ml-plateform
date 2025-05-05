#!/usr/bin/env bash
set -e          # stop on first error
set -o pipefail # catch errors in pipelines

###############################################################################
# Colours & helpers
###############################################################################
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No colour

print_message() {  # $1 = text, $2 = colour
    echo -e "${2}${1}${NC}"
}

command_exists() { command -v "$1" >/dev/null 2>&1; }

###############################################################################
# Banner
###############################################################################
print_message "=================================================" "$YELLOW"
print_message "First‑time setup – No‑Code ML Platform" "$YELLOW"
print_message "=================================================" "$YELLOW"
echo ""

###############################################################################
# Pre‑flight checks
###############################################################################
if ! command_exists conda; then
    print_message "Conda not found. Install Miniconda / Mambaforge first:" "$RED"
    print_message "https://docs.conda.io/en/latest/miniconda.html" "$YELLOW"
    exit 1
fi

# Make sure the conda shell hook is available even inside scripts
source "$(conda info --base)/etc/profile.d/conda.sh"

###############################################################################
# Create or recreate the Conda environment
###############################################################################
ENV_NAME="nocodeml"
PY_VER="3.10"

if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    print_message "Removing existing Conda env '${ENV_NAME}'…" "$YELLOW"
    conda env remove -n "${ENV_NAME}" -y
fi

print_message "Creating Conda env '${ENV_NAME}' (Python ${PY_VER})…" "$YELLOW"
conda create -n "${ENV_NAME}" python="${PY_VER}" -y

###############################################################################
# Activate the env & install dependencies
###############################################################################
print_message "Activating env…" "$YELLOW"
conda activate "${ENV_NAME}"   # works because of the earlier ‘source …/conda.sh’

print_message "Upgrading pip / build tools…" "$YELLOW"
python -m pip install --upgrade pip setuptools wheel

print_message "Installing requirements from requirements.txt…" "$YELLOW"
pip install -r requirements.txt

###############################################################################
# Smoke‑test key libraries
###############################################################################
print_message "Verifying installations…" "$YELLOW"
python - <<'PYTEST'
import importlib, sys
for pkg in ("streamlit", "pandas", "numpy", "sklearn", "plotly",
            "matplotlib", "seaborn"):
    if importlib.util.find_spec(pkg) is None:
        sys.exit(f"Missing package: {pkg}")
print("All required packages import correctly.")
PYTEST
print_message "Package check passed." "$GREEN"

###############################################################################
# Generate the run script
###############################################################################
print_message "Creating start_app.sh…" "$YELLOW"
cat > start_app.sh <<'EOS'
#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
print_message() { echo -e "${2}${1}${NC}"; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nocodeml

print_message "Starting No‑Code ML Platform…" "$GREEN"
print_message "Opening your default browser – press Ctrl+C to quit." "$GREEN"

streamlit run app.py --server.enableXsrfProtection false
EOS
chmod +x start_app.sh

###############################################################################
# Finished
###############################################################################
print_message "=================================================" "$GREEN"
print_message "Setup completed successfully!" "$GREEN"
print_message "Run ./start_app.sh to launch the application." "$GREEN"
print_message "=================================================" "$GREEN"


#!/bin/bash
set -e

echo "========================================"
echo " BodySim OSP Environment Setup"
echo "========================================"

# ── Step 1: System dependencies + R ──────────────────────────────────
echo "[1/5] Installing system dependencies and R..."
sudo apt-get install -y -qq \
  r-base r-base-dev \
  libcurl4-openssl-dev libssl-dev libxml2-dev \
  libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
  libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
  libuv1-dev \
  libsodium-dev
sudo apt-get update -qq
sudo apt-get install -y -qq \
  r-base r-base-dev \
  libcurl4-openssl-dev libssl-dev libxml2-dev \
  libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
  libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
  libuv1-dev
echo "      R $(R --version | head -1 | cut -d' ' -f3) installed."

# ── Step 2: .NET 8 (required by rSharp/.NET bridge in ospsuite) ───────
echo "[2/5] Installing .NET 8..."
wget -q https://dot.net/v1/dotnet-install.sh -O /tmp/dotnet-install.sh
chmod +x /tmp/dotnet-install.sh
/tmp/dotnet-install.sh --channel 8.0 --install-dir $HOME/.dotnet
export DOTNET_ROOT=$HOME/.dotnet
export PATH=$PATH:$HOME/.dotnet
echo 'export DOTNET_ROOT=$HOME/.dotnet' >> ~/.bashrc
echo 'export PATH=$PATH:$HOME/.dotnet' >> ~/.bashrc
echo "      .NET $(dotnet --version) installed."

# ── Step 3: Core Python packages ──────────────────────────────────────
echo "[3/5] Installing core Python packages..."
pip install -q \
  numpy scipy pandas rdkit \
  requests fastapi uvicorn \
  pydantic tqdm rich \
  rpy2==3.6.7
echo "      Done."

# ── Step 4: R packages (plumber + ospsuite) ───────────────────────────
# ospsuite-R is the official R binding for the PK-Sim simulation engine.
# It bundles the OSP core — no separate PK-Sim GUI installation needed on Linux.
# Requires: .NET 8 (Step 2), libuv1-dev (Step 1), GITHUB_PAT to avoid rate limits.
echo "[4/5] Installing R packages (10-15 min — compiling from source)..."
sudo DOTNET_ROOT=$HOME/.dotnet \
     PATH=$PATH:$HOME/.dotnet \
     GITHUB_PAT=$GITHUB_TOKEN \
Rscript -e "
Sys.setenv(GITHUB_PAT  = Sys.getenv('GITHUB_PAT'))
Sys.setenv(DOTNET_ROOT = Sys.getenv('DOTNET_ROOT'))
options(repos = 'https://cloud.r-project.org')
install.packages(c('remotes', 'plumber', 'jsonlite', 'fs'), quiet = TRUE)
cat('  base R packages: OK\n')
remotes::install_github('Open-Systems-Pharmacology/OSPSuite-R@*release', quiet = FALSE)
cat('  ospsuite-R: OK\n')
"

# ── Step 5: Set PKSIM_INSTALLATION_DIR ────────────────────────────────
# On Linux, ospsuite-R bundles its own engine data.
# PKSIM_INSTALLATION_DIR points to the R package data directory.
echo "[5/5] Configuring PKSIM_INSTALLATION_DIR..."
OSP_PATH=$(Rscript -e "cat(system.file(package='ospsuite'))" 2>/dev/null || echo "")
if [ -n "$OSP_PATH" ]; then
  export PKSIM_INSTALLATION_DIR=$OSP_PATH
  echo "export PKSIM_INSTALLATION_DIR=$OSP_PATH" >> ~/.bashrc
  echo "      PKSIM_INSTALLATION_DIR=$OSP_PATH"
else
  echo "      WARNING: ospsuite package path not found — set PKSIM_INSTALLATION_DIR manually."
fi

# ── Verification ──────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Verifying installation..."
echo "========================================"

echo -n "Python packages: "
python3 -c "import numpy, scipy, pandas, rdkit, rpy2; print('OK')"

echo -n ".NET runtime:    "
dotnet --version

echo -n "ospsuite-R:      "
Rscript -e "library(ospsuite); cat('v', as.character(packageVersion('ospsuite')), '\n', sep='')"

echo -n "plumber:         "
Rscript -e "library(plumber); cat('v', as.character(packageVersion('plumber')), '\n', sep='')"

echo ""
echo "========================================"
echo " Setup complete."
echo "========================================"
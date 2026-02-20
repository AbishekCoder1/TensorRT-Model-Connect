#!/usr/bin/env bash
# Sets up a fresh GB300 host for Claude Code + trtf development.
# Run once on each new GB300 machine (on the HOST, not inside Docker).
#
# What it does:
#   1. Creates local workspace dirs on NVMe (avoids NFS/autofs + bwrap conflict)
#   2. Installs nvm + Node.js + Claude Code
#   3. Clones the repo
#   4. Builds + starts the Docker dev container
#
# Usage:
#   ssh <gb300-host>
#   bash /home/yifeif/trt-transformers-cpp/scripts/setup_host_gb300.sh
#   # Then start a new shell and run: claude
set -euo pipefail

LOCAL_HOME="/workspace/users/yifeif"
REPO_DIR="${LOCAL_HOME}/trt-transformers-cpp"

echo "=== 1. Creating local workspace directories ==="
mkdir -p "${LOCAL_HOME}"/{.claude/debug,.npm/_logs,.nvm}

echo "=== 2. Installing nvm + Node.js ==="
export NVM_DIR="${LOCAL_HOME}/.nvm"
if [ ! -s "${NVM_DIR}/nvm.sh" ]; then
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
fi
# shellcheck source=/dev/null
source "${NVM_DIR}/nvm.sh"
nvm install --lts
nvm use --lts

echo "=== 3. Installing Claude Code ==="
npm install -g @anthropic-ai/claude-code

echo "=== 4. Cloning repo (if needed) ==="
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone git@github.com:nvidia/trt-transformers-cpp.git "${REPO_DIR}"
else
  echo "Repo already exists at ${REPO_DIR}, skipping clone."
fi

echo "=== 5. Building Docker dev container ==="
cd "${REPO_DIR}"
if docker image inspect trtf-dev-gb300 &>/dev/null; then
  echo "Docker image trtf-dev-gb300 already exists, skipping build."
else
  ./scripts/docker_build_gb300.sh
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "The .bashrc already sets HOME=${LOCAL_HOME} (shared via NFS)."
echo "Start a new shell, then run:"
echo "  claude                      # Claude Code with working sandbox"
echo "  ./scripts/docker_run_gb300.sh  # launch the dev container"

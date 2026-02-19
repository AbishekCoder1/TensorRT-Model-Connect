#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
docker build -t trtf-dev-gb300 -f Dockerfile.gb300 .

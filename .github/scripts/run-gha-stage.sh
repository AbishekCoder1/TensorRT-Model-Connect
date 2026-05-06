#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: run-gha-stage.sh <stage>}"

docker image inspect "$TRTMC_CI_IMAGE" >/dev/null || {
  echo "::error::Docker image '$TRTMC_CI_IMAGE' is not present on the self-hosted runner. Set repository variable TRTMC_CI_IMAGE if the runner uses a different local image tag."
  exit 1
}

extra_mounts=()
if [ -d /workspace/users/yifeif ]; then
  extra_mounts+=(-v /workspace/users/yifeif:/workspace/users/yifeif)
fi

# shellcheck disable=SC2086
docker run --rm \
  $TRTMC_CONTAINER_OPTIONS \
  "${extra_mounts[@]}" \
  -v "$GITHUB_WORKSPACE:$GITHUB_WORKSPACE" \
  -w "$GITHUB_WORKSPACE" \
  -e CI_BASE_REF \
  -e ENGINE_DIR \
  -e FULL_E2E \
  -e RUN_COVERAGE_MAP \
  -e REBUILD_ENGINES \
  -e GITHUB_EVENT_NAME \
  -e GITHUB_REF_NAME \
  -e GITHUB_RUN_ID \
  -e PYTHONHASHSEED \
  -e PYTHON_COVERAGE_MIN_LINE \
  -e PYTHON_COVERAGE_MIN_BRANCH \
  -e CPP_COVERAGE_MIN_LINE \
  -e CPP_COVERAGE_MIN_FUNCTION \
  -e CPP_COVERAGE_MIN_BRANCH \
  -e BUILD_ALL_TIMEOUT \
  -e CPP_UNIT_TIMEOUT \
  -e PYTHON_BUILDER_TIMEOUT \
  -e CPP_COVERAGE_TIMEOUT \
  -e GRAPH_OP_TIMEOUT \
  -e SELECTIVE_E2E_TIMEOUT \
  -e FULL_E2E_TIMEOUT \
  -e COVERAGE_MAP_TIMEOUT \
  -e HF_TOKEN \
  -e HUGGING_FACE_HUB_TOKEN \
  "$TRTMC_CI_IMAGE" \
  bash .github/scripts/run-trtmc-ci.sh "$stage"

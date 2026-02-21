#!/usr/bin/env bash
# Production template for agent orchestration environment variables.
# Source this file, then fill in modality-specific strict parity hooks.

set -euo pipefail

# -----------------------------------------------------------------------------
# Container + repository context
# -----------------------------------------------------------------------------
export CONTAINER="${CONTAINER:-trtf-dev-gb300}"
export REPO_IN_CONTAINER="${REPO_IN_CONTAINER:-/workspace/trt-transformers-cpp}"

# -----------------------------------------------------------------------------
# Subagent execution
# -----------------------------------------------------------------------------
# Required placeholders:
#   {prompt_file} {worktree}
# Optional placeholders:
#   {task_json} {task_json_file} {worker_id}
if [[ -z "${AGENT_SUBAGENT_CMD_TEMPLATE:-}" ]]; then
  export AGENT_SUBAGENT_CMD_TEMPLATE='codex exec -C {worktree} --prompt-file {prompt_file}'
fi

# Optional scheduler policy overrides:
# export AGENT_CPU_SLOT_CAP=48
# export AGENT_RAM_RESERVE_GB=256
# export AGENT_GPU_SLOT_THRESHOLDS='220:3,160:2,100:1'
# export AGENT_MAX_WORKERS_PER_STRATEGY='diffusion=4,video_diffusion=2'
# export AGENT_STRATEGY_MUTEXES='diffusion|video_diffusion'

# -----------------------------------------------------------------------------
# Strict parity hooks (fill these for non-decoder modalities).
# Templates are formatted with values such as:
#   {model} {bundle} {binary} {hf_python} {output_dir} {prompt}
# -----------------------------------------------------------------------------
# export TRTF_ENCODER_CPP_PARITY_CMD_TEMPLATE='python3 tools/parity/encoder_parity.py --model {model} --bundle {bundle} --binary {binary} --hf-python {hf_python}'
# export TRTF_ENCODER_DECODER_CPP_PARITY_CMD_TEMPLATE='python3 tools/parity/seq2seq_parity.py --model {model} --bundle {bundle} --binary {binary} --hf-python {hf_python}'
# export TRTF_VISION_ENCODER_CPP_PARITY_CMD_TEMPLATE='python3 tools/parity/vision_encoder_parity.py --model {model} --bundle {bundle} --binary {binary} --hf-python {hf_python}'
# export TRTF_AUDIO_ENCODER_CPP_PARITY_CMD_TEMPLATE='python3 tools/parity/audio_encoder_parity.py --model {model} --bundle {bundle} --binary {binary} --hf-python {hf_python}'
# export TRTF_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE='python3 tools/parity/diffusion_frame_parity.py --model {model} --bundle {bundle} --output-dir {output_dir} --binary {binary} --hf-python {hf_python} --prompt "{prompt}"'
# export TRTF_VIDEO_ENCODER_CPP_PARITY_CMD_TEMPLATE='python3 tools/parity/video_encoder_parity.py --model {model} --bundle {bundle} --binary {binary} --hf-python {hf_python}'
# export TRTF_VIDEO_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE='python3 tools/parity/video_diffusion_parity.py --model {model} --bundle {bundle} --output-dir {output_dir} --binary {binary} --hf-python {hf_python} --prompt "{prompt}"'
# export TRTF_TIME_SERIES_CPP_PARITY_CMD_TEMPLATE='python3 tools/parity/time_series_parity.py --model {model} --bundle {bundle} --binary {binary} --hf-python {hf_python}'
# export TRTF_RL_POLICY_CPP_PARITY_CMD_TEMPLATE='python3 tools/parity/rl_policy_parity.py --model {model} --bundle {bundle} --binary {binary} --hf-python {hf_python}'
# export TRTF_RL_VALUE_CPP_PARITY_CMD_TEMPLATE='python3 tools/parity/rl_value_parity.py --model {model} --bundle {bundle} --binary {binary} --hf-python {hf_python}'

# Required for VL validation.
# export TRTF_VL_IMAGE='/absolute/path/to/real_image.jpg'

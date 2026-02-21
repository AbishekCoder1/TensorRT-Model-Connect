#!/usr/bin/env bash
# Copy/paste into your shell and edit paths/commands as needed.

# -----------------------------------------------------------------------------
# Container + repository context
# -----------------------------------------------------------------------------
export CONTAINER=trtf-dev-gb300
export REPO_IN_CONTAINER=/workspace/trt-transformers-cpp

# -----------------------------------------------------------------------------
# Subagent execution command template
# Required placeholders: {prompt_file} {worktree}
# Optional placeholders: {task_json} {task_json_file} {worker_id}
# -----------------------------------------------------------------------------
export AGENT_SUBAGENT_CMD_TEMPLATE='codex exec -C {worktree} --prompt-file {prompt_file}'

# Optional scheduler policy overrides:
# export AGENT_CPU_SLOT_CAP=48
# export AGENT_RAM_RESERVE_GB=256
# export AGENT_GPU_SLOT_THRESHOLDS='220:3,160:2,100:1'
# export AGENT_MAX_WORKERS_PER_STRATEGY='diffusion=4,video_diffusion=2'
# export AGENT_STRATEGY_MUTEXES='diffusion|video_diffusion'

# -----------------------------------------------------------------------------
# Optional strict parity hooks by modality
# Fill these with your local parity commands for each modality.
# -----------------------------------------------------------------------------
# export TRTF_ENCODER_CPP_PARITY_CMD_TEMPLATE='python3 <path_to_encoder_parity.py> --model {model} --bundle {bundle}'
# export TRTF_ENCODER_DECODER_CPP_PARITY_CMD_TEMPLATE='python3 <path_to_seq2seq_parity.py> --model {model} --bundle {bundle}'
# export TRTF_VISION_ENCODER_CPP_PARITY_CMD_TEMPLATE='python3 <path_to_vision_encoder_parity.py> --model {model} --bundle {bundle}'
# export TRTF_AUDIO_ENCODER_CPP_PARITY_CMD_TEMPLATE='python3 <path_to_audio_encoder_parity.py> --model {model} --bundle {bundle}'
# export TRTF_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE='python3 <path_to_diffusion_frame_parity.py> --bundle {bundle} --out {output_dir} --prompt "{prompt}"'
# export TRTF_VIDEO_ENCODER_CPP_PARITY_CMD_TEMPLATE='python3 <path_to_video_encoder_parity.py> --model {model} --bundle {bundle}'
# export TRTF_VIDEO_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE='python3 <path_to_video_diffusion_parity.py> --bundle {bundle} --out {output_dir} --prompt "{prompt}"'
# export TRTF_TIME_SERIES_CPP_PARITY_CMD_TEMPLATE='python3 <path_to_time_series_parity.py> --model {model} --bundle {bundle}'
# export TRTF_RL_POLICY_CPP_PARITY_CMD_TEMPLATE='python3 <path_to_rl_policy_parity.py> --model {model} --bundle {bundle}'
# export TRTF_RL_VALUE_CPP_PARITY_CMD_TEMPLATE='python3 <path_to_rl_value_parity.py> --model {model} --bundle {bundle}'

# Real image file used by VL checks.
# export TRTF_VL_IMAGE=/absolute/path/to/sample_image.jpg

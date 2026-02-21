#!/usr/bin/env bash
# Smoke-test environment values for validating template plumbing.
# Do not use this in production.

set -euo pipefail

export AGENT_SUBAGENT_CMD_TEMPLATE='echo SUBAGENT_SMOKE prompt={prompt_file} worktree={worktree} worker={worker_id}'

export TRTF_ENCODER_CPP_PARITY_CMD_TEMPLATE='echo ENCODER_SMOKE model={model} bundle={bundle} binary={binary}'
export TRTF_ENCODER_DECODER_CPP_PARITY_CMD_TEMPLATE='echo ENCDEC_SMOKE model={model} bundle={bundle} binary={binary}'
export TRTF_VISION_ENCODER_CPP_PARITY_CMD_TEMPLATE='echo VISIONENC_SMOKE model={model} bundle={bundle} binary={binary}'
export TRTF_AUDIO_ENCODER_CPP_PARITY_CMD_TEMPLATE='echo AUDIOENC_SMOKE model={model} bundle={bundle} binary={binary}'
export TRTF_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE='echo DIFFUSION_SMOKE model={model} bundle={bundle} out={output_dir} prompt=\"{prompt}\"'
export TRTF_VIDEO_ENCODER_CPP_PARITY_CMD_TEMPLATE='echo VIDEOENC_SMOKE model={model} bundle={bundle} binary={binary}'
export TRTF_VIDEO_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE='echo VIDEODIFF_SMOKE model={model} bundle={bundle} out={output_dir} prompt=\"{prompt}\"'
export TRTF_TIME_SERIES_CPP_PARITY_CMD_TEMPLATE='echo TS_SMOKE model={model} bundle={bundle} binary={binary}'
export TRTF_RL_POLICY_CPP_PARITY_CMD_TEMPLATE='echo RL_POLICY_SMOKE model={model} bundle={bundle} binary={binary}'
export TRTF_RL_VALUE_CPP_PARITY_CMD_TEMPLATE='echo RL_VALUE_SMOKE model={model} bundle={bundle} binary={binary}'

export TRTF_VL_IMAGE='/tmp/vl-smoke-image.jpg'

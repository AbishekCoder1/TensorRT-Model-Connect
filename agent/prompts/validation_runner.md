# Validation Task

You are executing strict parity validation for one model task.

## Task Context
- `task_id`: `{{TASK_ID}}`
- `task_type`: `{{TASK_TYPE}}`
- `model_id`: `{{MODEL_ID}}`
- `runtime_strategy`: `{{RUNTIME_STRATEGY}}`
- `modality`: `{{MODALITY}}`

## Goal
Confirm Python build + C++ runtime behavior matches Hugging Face behavior for this modality.

## Required Validation
- Decoder/MoE/SSM: logits battery + exact text parity.
- Encoder: encoder diff battery + strict C++ parity hook.
- Vision-language: `diff_vl.py` with real image.
- Diffusion: component checks + generated frame checks + frame parity hook.

If any check fails, include exact failing command and concise diagnosis.

## Full Task JSON
```json
{{TASK_JSON}}
```

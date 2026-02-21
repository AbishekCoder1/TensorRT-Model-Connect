from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

HOOK_VARS = [
    "TRTF_ENCODER_CPP_PARITY_CMD_TEMPLATE",
    "TRTF_ENCODER_DECODER_CPP_PARITY_CMD_TEMPLATE",
    "TRTF_VISION_ENCODER_CPP_PARITY_CMD_TEMPLATE",
    "TRTF_AUDIO_ENCODER_CPP_PARITY_CMD_TEMPLATE",
    "TRTF_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE",
    "TRTF_VIDEO_ENCODER_CPP_PARITY_CMD_TEMPLATE",
    "TRTF_VIDEO_DIFFUSION_FRAME_PARITY_CMD_TEMPLATE",
    "TRTF_TIME_SERIES_CPP_PARITY_CMD_TEMPLATE",
    "TRTF_RL_POLICY_CPP_PARITY_CMD_TEMPLATE",
    "TRTF_RL_VALUE_CPP_PARITY_CMD_TEMPLATE",
]


def _fmt(name: str, template: str, values: dict[str, str]) -> str:
    try:
        return template.format(**values)
    except Exception as exc:
        raise RuntimeError(f"{name}: failed to format template: {exc}") from exc


def _run_shell(cmd: str) -> tuple[int, str, str]:
    proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate agent template env vars")
    parser.add_argument("--execute", action="store_true", help="Execute rendered commands")
    parser.add_argument("--worktree", default="/tmp/agent-worktree")
    parser.add_argument("--prompt-file", default="/tmp/agent-prompt.md")
    parser.add_argument("--task-json-file", default="/tmp/agent-task.json")
    parser.add_argument("--bundle", default="/tmp/agent-model.trtfb")
    parser.add_argument("--binary", default="./build/trtf")
    parser.add_argument("--hf-python", default=".venv/bin/python")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="/tmp/agent-diffusion")
    parser.add_argument("--prompt", default="Template smoke prompt")
    args = parser.parse_args()

    values = {
        "prompt_file": args.prompt_file,
        "worktree": args.worktree,
        "task_json": '{"id":"template-smoke"}',
        "task_json_file": args.task_json_file,
        "worker_id": "template-smoke-worker",
        "model": args.model,
        "bundle": args.bundle,
        "binary": args.binary,
        "hf_python": args.hf_python,
        "output_dir": args.output_dir,
        "prompt": args.prompt,
        "runtime": "decoder_kv_cache",
        "max_new_tokens": "8",
    }

    Path(args.prompt_file).write_text("# template smoke\n", encoding="utf-8")
    Path(args.task_json_file).write_text(values["task_json"] + "\n", encoding="utf-8")

    subagent_tpl = os.environ.get("AGENT_SUBAGENT_CMD_TEMPLATE", "").strip()
    if not subagent_tpl:
        raise SystemExit("AGENT_SUBAGENT_CMD_TEMPLATE is not set")

    rendered = _fmt("AGENT_SUBAGENT_CMD_TEMPLATE", subagent_tpl, values)
    print(f"AGENT_SUBAGENT_CMD_TEMPLATE -> {rendered}")
    if args.execute:
        rc, out, err = _run_shell(rendered)
        print(f"  rc={rc}")
        if out:
            print(f"  stdout={out}")
        if err:
            print(f"  stderr={err}")
        if rc != 0:
            raise SystemExit("subagent template command failed")

    for env_var in HOOK_VARS:
        tpl = os.environ.get(env_var, "").strip()
        if not tpl:
            print(f"{env_var} -> <unset> (skipped)")
            continue
        rendered = _fmt(env_var, tpl, values)
        print(f"{env_var} -> {rendered}")
        if args.execute:
            rc, out, err = _run_shell(rendered)
            print(f"  rc={rc}")
            if out:
                print(f"  stdout={out}")
            if err:
                print(f"  stderr={err}")
            if rc != 0:
                raise SystemExit(f"{env_var} command failed")

    vl_image = os.environ.get("TRTF_VL_IMAGE", "").strip()
    print(f"TRTF_VL_IMAGE -> {vl_image or '<unset>'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dispatch autopilot tasks to parallel Claude Code agents.

Takes the task list from discover.py and launches Claude Code sessions
across isolated agent workspaces. Each agent scaffolds a family plugin,
iterates until validation passes, then commits.

Usage:
    # Interactive (approve each batch)
    python3 scripts/autopilot/dispatch.py /tmp/gaps.json

    # Fully autonomous
    python3 scripts/autopilot/dispatch.py /tmp/gaps.json --mode auto

    # Dry run (print prompts, don't launch)
    python3 scripts/autopilot/dispatch.py /tmp/gaps.json --mode dry-run

    # Custom agent count
    python3 scripts/autopilot/dispatch.py /tmp/gaps.json --agents 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = "/workspace/users/yifeif/workspaces"
DEFAULT_AGENTS = ["agent-1", "agent-2", "agent-3", "agent-4"]

# The prompt is the ENTIRE automation. The LLM agent IS the worker.
WORKER_PROMPT = textwrap.dedent("""\
    You are an autonomous agent implementing a new model family for the trtmc
    framework. Work entirely inside the container. Do not ask questions — make
    decisions and proceed. If something fails, read the error, fix it, and retry.

    ## Task
    - model_type:  {model_type}
    - hf_id:       {hf_id}
    - family_name: {family_name}
    - container:   trtmc-dev-gb300-{agent_id}
    {trust_remote_code_line}

    ## Step 1: Scaffold the plugin
    Run inside the container:
    ```
    docker exec trtmc-dev-gb300-{agent_id} python3 scripts/new_family.py \\
        --model-type {model_type} --hf-repo {hf_id} --family-name {family_name}
    ```
    If the plugin file already exists, skip this step.

    ## Step 2: Validate
    Run inside the container:
    ```
    docker exec trtmc-dev-gb300-{agent_id} ./scripts/validate_family.sh {hf_id} \\
        --max-cache-length 256 {trust_flag}
    ```

    ## Step 3: Fix and retry (if validation failed)
    If validation fails:
    1. Read the error output carefully.
    2. Read the generated plugin at:
       tensorrt_model_connect/tensorrt_model_connect/families/{family_name}.py
    3. Read the HF model's source code to understand the architecture:
       - Look at the model's config.json for architecture details
       - If available, examine the HF modeling code to understand weight naming
       - Check weight key names by running:
         ```
         docker exec trtmc-dev-gb300-{agent_id} python3 -c "
         from safetensors import safe_open
         from huggingface_hub import snapshot_download
         import glob, os
         model_dir = snapshot_download('{hf_id}'{trust_remote_code_py})
         for f in sorted(glob.glob(os.path.join(model_dir, '*.safetensors'))):
             with safe_open(f, framework='pt') as sf:
                 for k in sf.keys():
                     print(f'{{k:60s}} {{list(sf.get_tensor(k).shape)}}')
         "
         ```
    4. Fix the plugin:
       - Fix weight key mappings in load_weights()
       - Fix build_engine() parameters (norm_type, mlp_type, position_type,
         activation) if needed
       - Handle fused QKV/MLP projections if the model uses them
       - Look at similar existing plugins for reference patterns
    5. Re-run validation (Step 2).
    Repeat up to 3 times. Each cycle: read error → understand → fix → validate.

    ## Step 4: Create E2E manifest (after validation passes)
    Write the file tests/e2e/models/{family_name}.json:
    ```json
    {{
        "name": "{family_name}",
        "hf_id": "{hf_id}",
        "bundle": "{family_name}.trtfb",
        "family": "{family_name}",
        "runtime_strategy": "decoder_kv_cache",
        "max_cache_length": 256,
        "prompt": "The capital of France is",
        "max_new_tokens": 20{trust_manifest_line}
    }}
    ```
    Adjust runtime_strategy if the model is not a standard decoder (check the
    plugin's runtime_strategy attribute if it has one).

    ## Step 5: Commit and push
    ```
    cd /workspace/users/yifeif/workspaces/{agent_id}/tensorrt-model-connect
    git checkout -b autopilot/{family_name}
    git add tensorrt_model_connect/tensorrt_model_connect/families/{family_name}.py
    git add tests/e2e/models/{family_name}.json
    git commit -m "feat: add {family_name} family plugin (autopilot)"
    git push -u origin autopilot/{family_name}
    ```

    ## Step 6: Write status
    At the end, write a JSON status file to the WORKSPACE (not container /tmp):
    ```
    echo '{{"family": "{family_name}", "status": "PASS", "branch": "autopilot/{family_name}"}}' \\
        > /workspace/users/yifeif/workspaces/{agent_id}/tensorrt-model-connect/.autopilot_status.json
    ```
    If all retries failed, write status "FAIL" with the last error message instead.
    Write this file on the HOST filesystem, not via docker exec.

    {optimize_section}

    ## Important rules
    - ALL commands run via `docker exec trtmc-dev-gb300-{agent_id}`.
    - Do NOT modify any file outside families/{family_name}.py and
      tests/e2e/models/{family_name}.json.
    - Do NOT edit shared files (checkpoint_mapper.py, standard_decoder_builder.py, etc.).
    - Look at existing plugins in families/ for reference when fixing issues.
    - If the model needs a completely new C++ runtime plugin (non-transformer
      architecture), write status "SKIP" with reason and stop.
""")


_OPTIMIZE_SECTION = """\
## Step 4b: Optimize precision (optional)

    After validation passes, optimize the model for low precision:
    1. Read the skill: cat /workspace/users/yifeif/workspaces/{agent_id}/tensorrt-model-connect/.claude/skills/optimize-model-precision.md
    2. Follow the skill to find the best non-FP32 precision config
    3. At minimum, build an FP16 variant
    4. Create a second E2E manifest for the optimized variant
"""


def build_prompt(task: dict, agent_id: str, *, optimize: bool = False) -> str:
    """Fill in the worker prompt template for a specific task."""
    trust_rc = task.get("trust_remote_code", False)
    optimize_section = ""
    if optimize:
        optimize_section = _OPTIMIZE_SECTION.format(
            agent_id=agent_id, family_name=task["family_name"])
    return WORKER_PROMPT.format(
        model_type=task["model_type"],
        hf_id=task["hf_id"],
        family_name=task["family_name"],
        agent_id=agent_id,
        trust_remote_code_line=(
            "- trust_remote_code: yes" if trust_rc else ""),
        trust_flag="--trust-remote-code" if trust_rc else "",
        trust_remote_code_py=", trust_remote_code=True" if trust_rc else "",
        trust_manifest_line=',\n    "trust_remote_code": true' if trust_rc else "",
        optimize_section=optimize_section,
    )


def find_claude_binary() -> str | None:
    """Find the claude CLI binary."""
    return shutil.which("claude")


def launch_agent(
    agent_id: str,
    task: dict,
    dry_run: bool = False,
    optimize: bool = False,
) -> subprocess.Popen | None:
    """Launch a Claude Code session for one task in one agent workspace."""
    workspace = f"{WORKSPACE_ROOT}/{agent_id}/tensorrt-model-connect"
    prompt = build_prompt(task, agent_id, optimize=optimize)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN: agent={agent_id}, family={task['family_name']}")
        print(f"Workspace: {workspace}")
        print(f"Prompt ({len(prompt)} chars):")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        return None

    claude_bin = find_claude_binary()
    if not claude_bin:
        print("ERROR: 'claude' CLI not found in PATH.", file=sys.stderr)
        sys.exit(1)

    # Launch claude in non-interactive mode
    # --print: non-interactive, print output
    # -p: pass prompt directly
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    log_path = os.path.join(tmpdir, f"autopilot_{task['family_name']}.log")
    log_file = open(log_path, "w")

    proc = subprocess.Popen(
        [
            claude_bin,
            "--print",                     # Non-interactive output mode
            "-p", prompt,                  # Direct prompt
            "--output-format", "text",
            "--max-turns", "30",           # Enough for scaffold + fix loops
            "--allowedTools", "Bash", "Read", "Write", "Edit", "Glob", "Grep",
        ],
        cwd=workspace,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    print(f"  Started agent {agent_id} (pid={proc.pid}, "
          f"family={task['family_name']}, log={log_path})")

    return proc


def wait_for_batch(
    procs: dict[str, tuple[subprocess.Popen, dict]],
    timeout: int = 1800,
) -> dict[str, dict]:
    """Wait for all processes in a batch to complete. Returns results."""
    results = {}
    start = time.time()

    while procs and (time.time() - start) < timeout:
        for agent_id, (proc, task) in list(procs.items()):
            ret = proc.poll()
            if ret is not None:
                family = task["family_name"]

                # Check status file (written to workspace by agent)
                status_file = (
                    f"{WORKSPACE_ROOT}/{agent_id}/"
                    f"tensorrt-model-connect/.autopilot_status.json"
                )
                status = {"family": family, "status": "UNKNOWN"}
                if os.path.exists(status_file):
                    try:
                        status = json.loads(Path(status_file).read_text())
                    except json.JSONDecodeError:
                        pass

                if ret == 0 and status.get("status") == "PASS":
                    print(f"  \u2713 {agent_id} ({family}): PASSED "
                          f"— branch {status.get('branch', '?')}")
                elif status.get("status") == "SKIP":
                    print(f"  - {agent_id} ({family}): SKIPPED "
                          f"— {status.get('reason', 'needs C++ plugin')}")
                else:
                    print(f"  \u2717 {agent_id} ({family}): FAILED "
                          f"(exit={ret})")

                results[agent_id] = status
                del procs[agent_id]
                break  # Restart loop after mutation

        if procs:
            time.sleep(5)

    # Handle timeouts
    for agent_id, (proc, task) in procs.items():
        proc.terminate()
        print(f"  ! {agent_id} ({task['family_name']}): TIMEOUT")
        results[agent_id] = {
            "family": task["family_name"], "status": "TIMEOUT"}

    return results


def dispatch(
    tasks: list[dict],
    agent_ids: list[str],
    mode: str = "interactive",
    timeout: int = 1800,
):
    """Dispatch tasks across agent slots in batches."""
    batch_size = len(agent_ids)
    total_batches = (len(tasks) + batch_size - 1) // batch_size

    all_results = {}
    passed = failed = skipped = 0

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch = tasks[batch_start:batch_start + batch_size]

        print(f"\nBatch {batch_num + 1}/{total_batches} "
              f"({len(batch)} tasks):")
        for i, task in enumerate(batch):
            agent = agent_ids[i]
            print(f"  {agent}: {task['model_type']:<20} "
                  f"({task['hf_id']})")

        # Approval gate
        if mode == "interactive":
            try:
                answer = input("\nLaunch this batch? [Y/n/skip/quit] ").strip()
            except EOFError:
                answer = "quit"
            if answer.lower() in ("n", "no", "skip"):
                print("Skipping batch.")
                continue
            if answer.lower() in ("q", "quit"):
                print("Quitting.")
                break

        # Launch agents
        is_dry_run = mode == "dry-run"
        procs = {}
        for i, task in enumerate(batch):
            agent = agent_ids[i]
            proc = launch_agent(agent, task, dry_run=is_dry_run,
                                optimize=args.optimize)
            if proc:
                procs[agent] = (proc, task)

        if is_dry_run:
            continue

        # Wait for batch
        print(f"\nWaiting for batch {batch_num + 1} "
              f"(timeout={timeout}s)...")
        results = wait_for_batch(procs, timeout=timeout)
        all_results.update(results)

        for r in results.values():
            s = r.get("status", "UNKNOWN")
            if s == "PASS":
                passed += 1
            elif s == "SKIP":
                skipped += 1
            else:
                failed += 1

    # Final summary
    print(f"\n{'='*50}")
    print(f"  Autopilot Summary")
    print(f"{'='*50}")
    print(f"  Tasks:   {len(tasks)}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"{'='*50}")

    # Save full results
    results_path = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "autopilot_results.json")
    Path(results_path).write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dispatch autopilot tasks to parallel Claude Code agents.")
    parser.add_argument("tasks_file", help="JSON file from discover.py")
    parser.add_argument("--agents", type=int, default=4,
                        help="Number of parallel agents (default: 4)")
    parser.add_argument("--agent-ids", nargs="+", default=None,
                        help="Specific agent IDs (default: agent-1..N)")
    parser.add_argument("--mode", choices=["interactive", "auto", "dry-run"],
                        default="interactive",
                        help="Execution mode (default: interactive)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-batch timeout in seconds (default: 1800)")
    parser.add_argument("--optimize", action="store_true",
                        help="After validation, run precision optimization")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N tasks (for resuming)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max tasks to process")
    args = parser.parse_args()

    # Load tasks
    data = json.loads(Path(args.tasks_file).read_text())
    tasks = data["tasks"] if isinstance(data, dict) else data
    tasks = tasks[args.offset:]
    if args.limit:
        tasks = tasks[:args.limit]

    if not tasks:
        print("No tasks to dispatch.")
        return

    # Agent IDs
    agent_ids = args.agent_ids or [f"agent-{i+1}" for i in range(args.agents)]

    print(f"Autopilot: {len(tasks)} tasks across {len(agent_ids)} agents")
    print(f"Mode: {args.mode}")

    # Verify agent workspaces exist
    for aid in agent_ids:
        ws = Path(f"{WORKSPACE_ROOT}/{aid}/tensorrt-model-connect")
        if not ws.is_dir() and args.mode != "dry-run":
            print(f"WARNING: Workspace {ws} does not exist.", file=sys.stderr)
            print(f"  Bootstrap it: ./scripts/bootstrap_workspace.sh "
                  f"--id {aid} --branch master --detach", file=sys.stderr)

    dispatch(tasks, agent_ids, mode=args.mode, timeout=args.timeout)


if __name__ == "__main__":
    main()

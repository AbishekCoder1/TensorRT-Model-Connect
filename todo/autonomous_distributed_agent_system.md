# Autonomous Distributed Agent System for trt-transformers-cpp

## Context

Currently, Claude Code works on a single task then stops, requiring manual re-engagement for each task. With 60+ model families to integrate and multiple infrastructure work streams, this creates a massive scaling bottleneck. The goal is an autonomous **distributed** system where multiple Claude Code agents run continuously across **multiple physical machines** — discovering tasks, implementing, validating, committing, and moving to the next task — with minimal human intervention.

Inspired by [Anthropic's C compiler project](https://www.anthropic.com/engineering/building-c-compiler) which used 16 parallel Claude agents with infinite loops, git-based task locking, and a high-quality test harness as the oracle.

**Existing foundation**: `scripts/launch_model_agents.py` + `scripts/agents/implement-model-family.md` already implement a single-shot version of this for the OLD C++ architecture. We're upgrading this to the current Python-build architecture and making it fully autonomous and distributed.

---

## Distributed Architecture Overview

```
                    ┌─────────────────────┐
                    │   Git Remote (origin)│
                    │   (GitHub / GitLab)  │
                    │                      │
                    │  agent/backlog.json   │  ← Source of truth
                    │  agent/claimed/*.lock │  ← Distributed locks
                    │  agent/completed/*    │  ← Results
                    │  agent/failed/*       │  ← Failures
                    └──────┬──────┬────────┘
                           │      │
              ┌────────────┘      └────────────┐
              │                                │
     ┌────────▼─────────┐           ┌──────────▼────────┐
     │  Machine A (GPU)  │           │  Machine B (GPU)   │
     │  git clone + venv │           │  git clone + venv  │
     │                   │           │                    │
     │  agent_loop.sh    │           │  agent_loop.sh     │
     │  ├─ agent-a-1     │           │  ├─ agent-b-1      │
     │  └─ agent-a-2     │           │  └─ agent-b-2      │
     │                   │           │                    │
     │  Docker container │           │  Docker container  │
     │  (GPU validation) │           │  (GPU validation)  │
     └───────────────────┘           └────────────────────┘
              │                                │
              └───── git push/pull ────────────┘
                  (coordination channel)
```

**Key principle**: Git remote is the **only** coordination channel. Each machine has its own independent checkout. Agents on different machines never share a filesystem — they coordinate entirely through `git push/pull` of the `agent/` directory.

---

## System Directory Structure

```
agent/
├── backlog.json              # Task queue (priority-ordered, machine-readable)
├── claimed/                  # Lock files: <task-id>.lock (contains agent ID + timestamp + machine)
├── completed/                # Finished task records with results
├── failed/                   # Failed task records with diagnosis
├── logs/                     # Per-agent session logs (local only, .gitignored)
├── prompts/                  # Generated per-task prompts (local only, .gitignored)
├── .gitignore                # Ignore logs/ and prompts/ (local-only dirs)
├── AGENT_PROMPT.md           # Master prompt template (the "brain")
├── agent_loop.sh             # Infinite loop driver script
├── claim_task.py             # Atomic task claiming via git push race
├── generate_prompt.py        # Compose base + role + task → concrete prompt
├── report_status.py          # Dashboard: what's running, what's done, what's stuck
├── setup_machine.sh          # Per-machine one-time setup (clone, venv, container)
└── roles/                    # Specialized role prompt overlays
    ├── backlog-generator.md  # Autonomous task discovery & backlog population
    ├── model-integrator.md   # Standard family plugin integration
    ├── infra-builder.md      # New graph ops, runtime backends
    ├── regression-runner.md  # Continuous test suite runner
    └── docs-updater.md       # Keep docs/wiki + WORKLOG in sync
```

---

## 1. Backlog Generator Agent (autonomous task discovery)

Instead of a static script, the **backlog-generator** is itself an agent role that runs continuously. It:

1. **Scans HuggingFace Hub** for popular models not yet supported
2. **Downloads config.json** from candidate models, classifies by architecture
3. **Matches against existing plugins** — if model_type already matched by a plugin, skip
4. **Triages complexity** — standard decoder vs. fused weights vs. new architecture
5. **Appends new tasks** to `agent/backlog.json` with appropriate priority and metadata
6. **Runs periodically** (every few hours, or after all current tasks complete)

### Backlog task format

```json
{
  "id": "family-yi",
  "type": "model-integration",
  "role": "model-integrator",
  "priority": 1,
  "tier": "P1",
  "title": "Add Yi family plugin",
  "hf_repo": "01-ai/Yi-1.5-6B",
  "model_type": "yi",
  "architectures": ["YiForCausalLM"],
  "runtime_strategy": "decoder_kv_cache",
  "complexity": "standard-decoder",
  "estimated_minutes": 15,
  "blocked_by": [],
  "description": "Standard decoder, LLaMA-like. Python plugin only.",
  "added_by": "backlog-generator",
  "added_at": "2026-02-18T10:00:00"
}
```

### Backlog generator prompt (`agent/roles/backlog-generator.md`)

The agent receives:
- Current `backlog.json` (what's already planned)
- Current list of `trtf_build/trtf_build/families/*.py` (what's already supported)
- `todo/scaling_gap_analysis.md` + `todo/hf_model_coverage_gap_analysis.md` (strategic plans)
- Instructions to:
  1. Parse the gap analysis docs for explicitly listed target families
  2. Query HuggingFace Hub API for trending models (via `huggingface_hub` Python API)
  3. For each candidate: download `config.json`, check `model_type` and `architectures`
  4. Classify: does it match an existing plugin? → skip. Standard decoder? → P1 task. Fused weights? → P2. New architecture? → P5+.
  5. Append to `backlog.json`, commit, push

### Seeding the initial backlog

On first run, the backlog-generator agent reads the gap analysis docs and creates ~60 tasks from the already-identified families. Subsequent runs add newly-trending models.

---

## 2. Distributed Agent Loop (`agent/agent_loop.sh`)

The core infinite loop — designed to work on any machine with a git checkout:

```bash
#!/usr/bin/env bash
set -euo pipefail

AGENT_ID="${1:-agent-$(hostname)-$$}"
MACHINE_ID="$(hostname)"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/agent/logs"
mkdir -p "$LOG_DIR" "$REPO_ROOT/agent/prompts"

echo "[$AGENT_ID@$MACHINE_ID] Starting autonomous loop at $(date -Iseconds)"

while true; do
    # 1. Sync: pull latest backlog + claims from all machines
    cd "$REPO_ROOT"
    git stash --include-untracked 2>/dev/null || true
    git checkout master 2>/dev/null || true
    git pull --rebase origin master 2>/dev/null || true
    git stash pop 2>/dev/null || true

    # 2. Claim next available task (atomic: claim + commit + push)
    TASK_JSON=$("$REPO_ROOT/agent/claim_task.py" \
        --agent-id "$AGENT_ID" \
        --machine-id "$MACHINE_ID")

    if [ -z "$TASK_JSON" ]; then
        echo "[$AGENT_ID] No unclaimed tasks. Sleeping 60s..."
        sleep 60
        continue
    fi

    TASK_ID=$(echo "$TASK_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    TASK_ROLE=$(echo "$TASK_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['role'])")
    LOGFILE="$LOG_DIR/${AGENT_ID}_${TASK_ID}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$AGENT_ID@$MACHINE_ID] Claimed: $TASK_ID (role: $TASK_ROLE)"

    # 3. Generate task-specific prompt
    PROMPT_FILE="$REPO_ROOT/agent/prompts/${TASK_ID}.md"
    "$REPO_ROOT/agent/generate_prompt.py" \
        --task-id "$TASK_ID" \
        --role "$TASK_ROLE" \
        --agent-id "$AGENT_ID" \
        --output "$PROMPT_FILE"

    # 4. Create feature branch from latest master
    BRANCH="agent/${TASK_ID}"
    git checkout -b "$BRANCH" origin/master 2>/dev/null || git checkout "$BRANCH"

    # 5. Run Claude Code with the generated prompt
    claude --dangerously-skip-permissions \
           -p "$(cat "$PROMPT_FILE")" \
           --model claude-sonnet-4-6 \
           &> "$LOGFILE"
    EXIT_CODE=$?

    # 6. Report results back to remote
    git checkout master 2>/dev/null || true
    git pull --rebase origin master 2>/dev/null || true

    if [ $EXIT_CODE -eq 0 ]; then
        # Mark completed
        python3 -c "
import json, time
from pathlib import Path
lock = Path('agent/claimed/${TASK_ID}.lock')
if lock.exists():
    rec = json.loads(lock.read_text())
    rec['status'] = 'completed'
    rec['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    Path('agent/completed/${TASK_ID}.json').write_text(json.dumps(rec, indent=2))
    lock.unlink()
"
        git add agent/completed/ agent/claimed/ 2>/dev/null || true
        git commit -m "agent($TASK_ID): completed by $AGENT_ID@$MACHINE_ID" 2>/dev/null || true
        echo "[$AGENT_ID] COMPLETED: $TASK_ID"
    else
        # Mark failed
        python3 -c "
import json, time
from pathlib import Path
lock = Path('agent/claimed/${TASK_ID}.lock')
if lock.exists():
    rec = json.loads(lock.read_text())
    rec['status'] = 'failed'
    rec['exit_code'] = $EXIT_CODE
    rec['failed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    try:
        with open('$LOGFILE') as f:
            rec['tail'] = ''.join(f.readlines()[-50:])
    except: pass
    rec['retry_count'] = rec.get('retry_count', 0) + 1
    Path('agent/failed/${TASK_ID}.json').write_text(json.dumps(rec, indent=2))
    lock.unlink()
"
        git add agent/failed/ agent/claimed/ 2>/dev/null || true
        git commit -m "agent($TASK_ID): FAILED by $AGENT_ID@$MACHINE_ID" 2>/dev/null || true
        echo "[$AGENT_ID] FAILED: $TASK_ID (exit=$EXIT_CODE)"
    fi

    # 7. Push everything: feature branch + status updates on master
    git push origin master 2>/dev/null || {
        # Push conflict = another machine updated. Rebase and retry.
        git pull --rebase origin master 2>/dev/null && git push origin master 2>/dev/null || true
    }
    git push origin "$BRANCH" 2>/dev/null || true

    sleep 5
done
```

---

## 3. Distributed Task Claiming (`agent/claim_task.py`)

**Atomic claiming via git push race** — the key to distributed coordination:

```python
#!/usr/bin/env python3
"""
Atomic distributed task claiming.

The race condition is resolved by git push:
1. Agent writes lock file locally
2. Agent commits + pushes to remote
3. If push succeeds → claim is valid (first writer wins)
4. If push fails → another agent claimed first → revert + try next task
"""
import json, os, subprocess, sys, time
from pathlib import Path

def git(*args):
    r = subprocess.run(["git"] + list(args), capture_output=True, text=True)
    return r.returncode == 0

def main():
    args = sys.argv
    agent_id = args[args.index("--agent-id") + 1]
    machine_id = args[args.index("--machine-id") + 1] if "--machine-id" in args else "local"
    root = Path(__file__).parent

    backlog = json.loads((root / "backlog.json").read_text())
    claimed = {p.stem for p in (root / "claimed").glob("*.lock")}
    completed = {p.stem for p in (root / "completed").glob("*.json")}
    failed_dir = root / "failed"

    for task in sorted(backlog["tasks"], key=lambda t: t["priority"]):
        tid = task["id"]
        if tid in claimed or tid in completed:
            continue

        # Check retry limit for failed tasks
        fail_file = failed_dir / f"{tid}.json"
        if fail_file.exists():
            rec = json.loads(fail_file.read_text())
            if rec.get("retry_count", 0) >= 3:
                continue

        # Check dependencies
        if any(dep not in completed for dep in task.get("blocked_by", [])):
            continue

        # Attempt atomic claim: write + commit + push
        lock = root / "claimed" / f"{tid}.lock"
        lock.write_text(json.dumps({
            "task": task,
            "agent_id": agent_id,
            "machine_id": machine_id,
            "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "in_progress"
        }, indent=2))

        # Atomic push — if another machine claimed it, push fails
        repo_root = root.parent
        os.chdir(repo_root)
        git("add", str(lock))
        git("commit", "-m", f"agent: {agent_id}@{machine_id} claims {tid}")

        if git("push", "origin", "master"):
            # Push succeeded — we own this task
            print(json.dumps(task))
            return
        else:
            # Push failed — another machine got it first
            git("reset", "HEAD~1")  # undo our commit
            lock.unlink(missing_ok=True)
            git("pull", "--rebase", "origin", "master")
            # Update our view of claims and try next task
            claimed = {p.stem for p in (root / "claimed").glob("*.lock")}
            continue

    print("")  # nothing available

if __name__ == "__main__":
    main()
```

---

## 4. Per-Machine Setup (`agent/setup_machine.sh`)

Run once on each new machine to join the agent fleet:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:?Usage: setup_machine.sh <git-remote-url> [storage-path]}"
STORAGE="${2:-/mnt/storage/trt-transformers}"

echo "=== Setting up agent machine: $(hostname) ==="

# 1. Clone repo
if [ ! -d trt-transformers-cpp ]; then
    git clone "$REPO_URL" trt-transformers-cpp
fi
cd trt-transformers-cpp

# 2. Ensure agent directories exist
mkdir -p agent/{claimed,completed,failed,logs,prompts}

# 3. Build and start container
./scripts/docker_build.sh
./scripts/docker_run.sh  # mounts $STORAGE automatically

# 4. Container setup (venv, deps, build)
docker exec trtf-dev ./scripts/setup_container.sh

# 5. Ensure storage dirs exist
mkdir -p "$STORAGE/engines" "$STORAGE/model-weights"

# 6. Verify
echo "=== Machine $(hostname) ready ==="
echo "Start agents with:"
echo "  bash agent/agent_loop.sh agent-$(hostname)-1"
echo "  bash agent/agent_loop.sh agent-$(hostname)-2  # if GPU allows"
```

---

## 5. Specialized Agent Roles

### `backlog-generator` (autonomous — runs on any machine, no GPU needed)
- Reads existing plugins, gap analysis docs, HF Hub trending
- Creates/updates `backlog.json` with new tasks
- Runs every few hours or when backlog is empty
- **No GPU required** — can run on a cheap CPU-only machine

### `model-integrator` (highest volume — ~60 tasks, GPU required)
- Creates Python family plugin in `trtf_build/trtf_build/families/`
- Adds E2E test manifest in `tests/e2e/models/`
- Runs `validate_family.sh` inside the Docker container
- Most tasks are Tier 1 (standard decoder) — ~15 min each

### `infra-builder` (rare but critical — ~10 tasks, GPU required)
- New graph ops in `graph_ops.py` / `graph_blocks.py`
- New runtime strategies (encoder-decoder, hybrid)
- Must pass full regression suite before merge

### `regression-runner` (continuous — always one running per machine, GPU required)
- Pulls completed branches, merges locally, runs Tier 4 full E2E
- Reports regressions to `agent/failed/`
- Acts as the "gatekeeper" before branches get merged to master

### `docs-updater` (periodic — no GPU needed)
- Updates `docs/wiki/` pages and `docs/WORKLOG.md`
- Runs after batches of model-integrator tasks complete

---

## 6. Verification Gates

Every task type has a mandatory verification gate:

| Role | Verification | Runs inside container? |
|------|-------------|----------------------|
| model-integrator | `validate_family.sh` (diff_logits + diff_layers + runner parity) | Yes (GPU) |
| infra-builder | Tier 1-3 regression (unit + graph ops + smoke) | Yes (GPU) |
| regression-runner | Tier 4 full E2E suite | Yes (GPU) |
| backlog-generator | JSON schema validation of backlog.json | No |
| docs-updater | Markdown lint + link check | No |

Critical design principle from the blog post: **"The task verifier is nearly perfect, otherwise Claude will solve the wrong problem."** Our verifier IS already perfect — `validate_family.sh` runs diff_logits + diff_layers + runner parity.

---

## 7. Coordination: Git Push Race Protocol

Cross-machine coordination uses **git push as an atomic lock**:

```
Machine A                         Git Remote                      Machine B
    │                                │                                │
    ├─ pull (see task unclaimed)     │                                │
    ├─ write claimed/task-1.lock     │                                │
    ├─ commit                        │                                │
    ├─ push ─────────────────────►   │ ◄─ push ──────────────────────┤ (simultaneous)
    │                           ACCEPTED (A wins)              REJECTED (B loses)
    │                                │                                │
    │                                │                   ├─ reset HEAD~1
    │                                │                   ├─ pull (see A's lock)
    │                                │                   ├─ pick next task
    │                                │                   └─ push → ACCEPTED
```

This is the same pattern the C compiler project used, extended across machines.

---

## 8. Status & Reporting

`agent/report_status.py` reads the git-tracked state directories:

```
$ python3 agent/report_status.py

=== Agent Fleet Status ===
Machines:  3 active (gpu-workstation-1, gpu-server-2, cpu-laptop-3)

Backlog:    47 tasks remaining (38 model-integration, 6 infra, 3 docs)
In-progress: 3 tasks
  agent-gpu1-a@gpu-workstation-1: family-yi         (claimed 5m ago)
  agent-gpu2-a@gpu-server-2:     family-cohere      (claimed 12m ago)
  agent-cpu-1@cpu-laptop-3:      backlog-generation  (claimed 2m ago)

Completed:  13 tasks (last: family-deepseek at 14:32 by gpu-server-2)
Failed:      1 tasks
  family-dbrx: build OOM (retry 1/3, last attempt by gpu-workstation-1)

Stale:       0 tasks (no locks older than 2h without update)
```

Any machine can run `report_status.py` after a `git pull` to see fleet-wide status.

---

## 9. Escalation Protocol

| Failure Type | Action | Max Retries |
|---|---|---|
| Build timeout | Increase `--max-cache-length` → retry | 2 |
| diff_logits tolerance fail | Loosen atol to 5e-3 → retry | 1 |
| Weight key mismatch | Analyze HF config, fix load_weights → retry | 3 |
| OOM during TRT build | Reduce cache length, add swap → retry | 1 |
| Unknown architecture | Write diagnosis to `agent/failed/`, skip | 0 |
| Regression in other model | Revert changes, write diagnosis, skip | 0 |

After max retries, task stays in `agent/failed/` with diagnosis for human review. The backlog-generator can also re-examine failed tasks and lower their priority or add new dependencies.

---

## 10. Bootstrap: Getting Started

### Step 1: First machine setup
```bash
# Clone the repo
git clone <repo-url> trt-transformers-cpp
cd trt-transformers-cpp

# Run per-machine setup
bash agent/setup_machine.sh <repo-url>
```

### Step 2: Seed the backlog (run backlog-generator once)
```bash
bash agent/agent_loop.sh backlog-gen-1
# Or manually: claude -p "$(cat agent/roles/backlog-generator.md)" --dangerously-skip-permissions
```

### Step 3: Start worker agents
```bash
# On machine 1 (GPU):
bash agent/agent_loop.sh agent-m1-a &
bash agent/agent_loop.sh agent-m1-b &  # if GPU memory allows 2 concurrent

# On machine 2 (GPU):
bash agent/agent_loop.sh agent-m2-a &

# On machine 3 (CPU-only, for backlog + docs):
bash agent/agent_loop.sh agent-m3-docs &
```

### Step 4: Add a new machine to the fleet
```bash
# On new machine:
bash agent/setup_machine.sh <repo-url> /mnt/data/trt-transformers
bash agent/agent_loop.sh agent-newmachine-a &
# That's it — it auto-discovers tasks via git pull
```

### Scaling
- **Scale up**: run `setup_machine.sh` on a new machine, start `agent_loop.sh`
- **Scale down**: kill the agent process (stale locks auto-cleaned after 2h)
- **GPU machines**: run model-integrator + regression-runner roles
- **CPU machines**: run backlog-generator + docs-updater roles

---

## Implementation Plan

### Files to create (in order)

1. **`agent/setup_machine.sh`** — Per-machine bootstrap (clone, container, venv)

2. **`agent/.gitignore`** — Ignore `logs/` and `prompts/` (local-only)

3. **`agent/claim_task.py`** — Distributed atomic task claiming via git push race

4. **`agent/roles/backlog-generator.md`** — Autonomous HF Hub scanner + task creator
   - Reads existing families via `pkgutil`-style scan of `trtf_build/trtf_build/families/`
   - Downloads `config.json` from HF Hub candidates
   - Classifies by complexity tier
   - Appends to `backlog.json`

5. **`agent/backlog.json`** — Initial seed (generated by first backlog-generator run, or hand-seeded with ~10 high-priority tasks from `scaling_gap_analysis.md`)

6. **`agent/AGENT_PROMPT.md`** — Master prompt template
   - Adapt from `scripts/agents/implement-model-family.md` for Python-build architecture
   - Reference `trtf_build/trtf_build/families/qwen.py` as standard pattern
   - Include inline examples of `FamilyPlugin` protocol from `families/base.py`

7. **`agent/roles/model-integrator.md`** — Step-by-step: create plugin → add manifest → validate_family.sh → commit

8. **`agent/roles/infra-builder.md`** — For new graph ops and runtime backends

9. **`agent/roles/regression-runner.md`** — Continuous E2E test runner

10. **`agent/roles/docs-updater.md`** — Documentation sync after task batches

11. **`agent/generate_prompt.py`** — Compose base + role + task → concrete prompt

12. **`agent/agent_loop.sh`** — The distributed infinite loop driver

13. **`agent/report_status.py`** — Fleet-wide status dashboard

### Files to modify

- **`CLAUDE.md`** — Add "Autonomous Agent System" section documenting `agent/`

### Verification

1. Create agent dirs + seed backlog with 3 test tasks
2. Run `claim_task.py` from two terminals simultaneously → verify only one wins per task
3. Generate prompt for a known-good model (Yi) → verify it's a complete, actionable prompt
4. Single-iteration test: run one loop iteration on one machine → verify claimed → completed flow
5. Two-machine test: run `agent_loop.sh` on two machines → verify they pick different tasks and don't collide
6. Full end-to-end: let the system run for 5 tasks → verify plugins created, manifests added, validation passed

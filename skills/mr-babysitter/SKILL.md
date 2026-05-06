---
name: mr-babysitter
description: Use when monitoring GitLab MR CI pipelines and fixing failures. Invoke via /loop for continuous monitoring, or manually to check and fix CI status across all open MRs.
---

# MR Babysitter

## Overview

Monitor all open non-draft GitLab MRs, detect CI failures, diagnose root causes from artifacts and logs, rebase when needed, fix code, and push — repeating until all pipelines are green. All work happens sequentially in a single agent workspace.

## Environment

- **Workspace:** `/workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect`
- **Container:** `trtmc-dev-gb300-agent-1`
- **GitLab project:** `yifeif/tensorrt-model-connect` on `gitlab-master.nvidia.com`
- **Project slug (URL-encoded):** `yifeif%2Ftensorrt-model-connect`

Ensure the container is running before any build/test commands:
```bash
docker start trtmc-dev-gb300-agent-1 2>/dev/null
```

## Workflow

Each invocation runs one full cycle. When used with `/loop`, this repeats automatically.

### Step 0: Lock guard

Before doing anything, check for a lock file to prevent overlapping cycles:

```bash
LOCK_FILE="/tmp/mr_babysitter.lock"

# Check if lock exists and is recent (less than 60 minutes old)
if [ -f "$LOCK_FILE" ]; then
  lock_age=$(( $(date +%s) - $(stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0) ))
  if [ "$lock_age" -lt 3600 ]; then
    echo "Previous cycle still running (lock age: ${lock_age}s). Skipping."
    exit 0
  fi
  # Stale lock (>60 min) — previous cycle likely crashed. Remove and continue.
fi

# Create lock
touch "$LOCK_FILE"
```

At the end of the cycle (after Step 4 report), remove the lock:
```bash
rm -f "$LOCK_FILE"
```

If the cycle exits early for any reason (error, nothing to do), also remove the lock.

### Step 1: Survey ALL open MRs

**CRITICAL: Dynamically discover MRs every cycle.** New MRs may appear, drafts may
be un-drafted, and MRs may be closed between cycles. NEVER hardcode MR IIDs.

Use a single script that fetches everything in one pass. Write a helper script to
`.ci_artifacts/survey.py` and run it:

```python
#!/usr/bin/env python3
"""Fetch all open non-draft MRs with pipeline status and behind-master count."""
import json, subprocess, sys

PROJECT = "yifeif%2Ftensorrt-model-connect"

def api(path):
    r = subprocess.run(
        ["glab", "api", f"projects/{PROJECT}/{path}"],
        capture_output=True, text=True)
    return json.loads(r.stdout.split("\n")[0]) if r.stdout.strip() else []

def behind_master(branch):
    r = subprocess.run(
        ["git", "rev-list", "--count", f"origin/{branch}..origin/master"],
        capture_output=True, text=True)
    return int(r.stdout.strip()) if r.returncode == 0 else -1

# 1. Fetch ALL open MRs
all_mrs = api("merge_requests?state=opened&author_username=yifeif&per_page=50")

# 2. Filter non-draft, get pipeline + behind count for each
results = []
for mr in all_mrs:
    if mr.get("draft", False):
        continue
    iid = mr["iid"]
    branch = mr["source_branch"]
    pipes = api(f"merge_requests/{iid}/pipelines?per_page=1")
    pipe_id = pipes[0]["id"] if pipes else None
    pipe_status = pipes[0]["status"] if pipes else "none"
    behind = behind_master(branch)
    results.append({
        "iid": iid, "branch": branch, "pipe_id": pipe_id,
        "pipe_status": pipe_status, "behind": behind,
        "title": mr["title"][:55],
        "merge_status": mr.get("detailed_merge_status", ""),
    })

# 3. Print dashboard
print(f"{'MR':<6}{'Status':<10}{'Behind':<8}{'Branch':<40}{'Title'}")
for r in results:
    print(f"!{r['iid']:<5}{r['pipe_status']:<10}{r['behind']:<8}{r['branch']:<40}{r['title']}")

# 4. Output JSON for next steps
json.dump(results, open(".ci_artifacts/survey_results.json", "w"), indent=2)
print(f"\nWrote {len(results)} MRs to .ci_artifacts/survey_results.json")
```

Run the survey, then read `survey_results.json` to build the action queue. This
guarantees every non-draft MR is discovered — no hardcoded IIDs.

### Step 2: Build action queue

For each non-draft MR, classify and queue:

| Condition | Action |
|-----------|--------|
| Pipeline status is `running` / `pending` | Skip — wait for it |
| Pipeline status is `success` and up-to-date with master | Skip — nothing to do |
| Pipeline status is `success` but behind master | Queue: REBASE |
| `detailed_merge_status` is `need_rebase` or `conflict` | Queue: REBASE |
| Pipeline status is `failed` and behind master | Queue: REBASE+DIAGNOSE+FIX |
| Pipeline status is `failed` and up-to-date with master | Queue: DIAGNOSE+FIX |
| No pipeline | Skip |

**Always rebase first if behind master.** Then diagnose and fix in the same cycle —
do NOT wait for the next CI round-trip (which wastes 2-3 hours). The only case
where you rebase without diagnosing is when the pipeline is already green.

Process the queue **sequentially**, one MR at a time.

### Step 3: Process each queued MR

For each MR in the queue:

#### 3a. Stash and checkout

```bash
cd /workspace/users/yifeif/workspaces/agent-1/tensorrt-model-connect
git stash --include-untracked
git fetch origin
git checkout {branch}
git pull origin {branch}
```

#### 3b. Rebase if behind master (always do this first)

If the branch is behind master, rebase before anything else.

```bash
git fetch origin master
git rebase origin/master
```

**Simple conflicts (same file, small edits):**
1. Read the conflicting files
2. Resolve conflicts intelligently (understand both sides)
3. `git add` resolved files
4. `git rebase --continue`
5. Repeat until rebase completes

**Complex conflicts (file renamed/deleted/split on master while MR modified it):**

The MR's functionality MUST be fully preserved. Never drop commits, skip conflicts,
or reduce scope. The goal is a mergeable branch with all original features intact.

1. Study the MR commits — understand what each one does (read the diffs, commit messages)
2. Study master's refactor — understand the new file structure and architecture
3. Re-apply the MR's intent onto the new architecture:
   - If a file was split into multiple files, apply relevant changes to each new file
   - If APIs changed, adapt the MR's code to the new interfaces
   - If names changed, update accordingly
4. Preserve all original commits (do not squash or drop during rebase)
5. Each rebase step should produce a compilable state if possible
6. After rebase completes, verify the full feature set still works

Then push:
```bash
git push origin {branch} --force-with-lease
```

#### 3c. Diagnose and fix (if pipeline was failed)

After rebasing (if needed), diagnose the failure and fix code in the same cycle.
Do NOT just push a rebase and wait — always diagnose+fix now.

##### Step 1: Identify which jobs failed

```bash
glab api "projects/yifeif%2Ftensorrt-model-connect/pipelines/{pipeline_id}/jobs"
```

##### Step 2: Get failure details

For **build failures** — read the job log:
```bash
glab api "projects/yifeif%2Ftensorrt-model-connect/jobs/{job_id}/trace"
```
Read the last 100-200 lines. Look for the actual `error:` line (ignore stb/third-party warnings).

For **E2E job failures** — download and parse artifacts:
```bash
TOKEN=$(glab auth status -t 2>&1 | awk '/Token found:/{print $NF}')
ARTIFACT_DIR=".ci_artifacts/mr{iid}"
mkdir -p "$ARTIFACT_DIR"
curl -s --header "PRIVATE-TOKEN: $TOKEN" \
  "https://gitlab-master.nvidia.com/api/v4/projects/yifeif%2Ftensorrt-model-connect/jobs/{job_id}/artifacts" \
  -o "$ARTIFACT_DIR/artifacts.zip"
unzip -o -q "$ARTIFACT_DIR/artifacts.zip" -d "$ARTIFACT_DIR/extracted"
```

Then parse each `result.json` using the `status` field (not `verdict`):
```python
# Key fields in result.json:
# - status: "pass" or "fail"
# - failure_type: "compare_fail", "build_fail", "runtime_error", etc.
# - stages.{stage_name}.status: per-stage status
# - stages.{stage_name}.message: error message (e.g., "TRT audio generation failed (rc=-6)")
# - stages.{stage_name}.metrics.{metric}.passed: per-metric pass/fail
# - stages.{stage_name}.metrics.{metric}.value: actual value
# - stages.{stage_name}.metrics.{metric}.threshold: required threshold
```

For each failing model, check these metrics:
- `logit_cosine_p5` < threshold → logit divergence (weight mapping or graph issue)
- `normalized_text_edit_distance` > threshold → wrong text output
- `token_agreement_rate` < threshold → token mismatch
- Stage `message` containing "failed (rc=" → runtime crash (SIGABRT, SIGSEGV, etc.)
- Stage `status: "error"` with no metrics → crash before comparison could run

For **unit test / other job failures** — read the job log trace.

##### Step 3: Classify failures as pre-existing vs MR-introduced

For E2E failures, compare against master's latest successful pipeline:
```bash
# Get master's latest successful E2E job
glab api "projects/yifeif%2Ftensorrt-model-connect/pipelines?ref=master&per_page=1&status=success"
# Download its artifacts and check the same failing models
```

| Model fails on MR | Model fails on master | Classification |
|---|---|---|
| Yes | Yes | Pre-existing — not caused by MR, skip |
| Yes | No | **MR regression — must fix** |
| Yes | No artifacts | Assume MR regression unless clearly infra |

##### Step 4: Decide what to fix

At this point, the branch is already rebased (if it was behind master).
Now decide what code changes are needed:

| Diagnosis | Action |
|-----------|--------|
| Build error caused by rebase (stale include paths, renamed APIs) | Fix the stale references to match new codebase |
| Build error in MR's own code | Fix the code |
| E2E failure that also fails on master | Skip (pre-existing, not caused by MR) |
| E2E failure only on this branch (MR regression) | **Fix the code** |
| E2E crash (rc=-6, SIGABRT, etc.) in MR-touched pipeline | **Fix the code — investigate MR's diff for the affected pipeline** |
| Unit test failure | Fix the code or test |
| Infrastructure failure (OOM, timeout, network) | Note and skip |

##### Step 5: Fix the code

Based on the diagnosis:
- **Family plugin weight mapping error:** Fix `tensorrt_model_connect/tensorrt_model_connect/families/{family}.py`
- **Runtime crash in a pipeline touched by MR:** Read the MR's diff for the affected
  pipeline. Look for buffer size mismatches, null pointer issues, incorrect tensor
  shapes/dtypes, missing engine bindings, or CUDA synchronization errors.
- **Threshold too tight:** Update the E2E manifest `tests/e2e/models/{model}.json`
  (only if values are very close to threshold AND the model also nearly-fails on master)
- **C++ compilation error:** Fix the relevant source in `src/`
- **Test code error:** Fix the test
- **Missing registration:** Check `cmake/trtmc_pipeline_plugins.cmake` and the plugin's registration macro

##### Step 6: Validate locally before pushing (Tier 1)

```bash
# Python unit tests
docker exec trtmc-dev-gb300-agent-1 /opt/venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# C++ unit tests (if C++ changed, rebuild first)
docker exec trtmc-dev-gb300-agent-1 cmake --build build -j
docker exec trtmc-dev-gb300-agent-1 ctest --test-dir build --output-on-failure

# Cyclomatic complexity (if C++ changed)
docker exec trtmc-dev-gb300-agent-1 python tools/check_cyclomatic_complexity.py src --max-ccn 10
```

##### Step 7: Commit and push

```bash
git add {specific files changed}
git commit -m "fix: {description of what was fixed and why}"
git push origin {branch}
```

#### 3d. Return to babysitter state

```bash
git checkout -  # back to previous branch
git stash pop   # restore any stashed work
```

### Step 4: Report

**Always print a full summary covering EVERY non-draft MR** — not just the ones you
acted on. The user needs a complete picture of all MR states at a glance.

```
=== Babysitter Cycle Summary ===
MR    Action      Branch                          Details
!79   WAITING     quantization-exploration         pipeline pending
!78   FIXED       coverage-based-unit-test        fixed missing import in test_selective.py
!77   DIAGNOSED   magpie-perf-optimization-v2     magpie-tts crash (rc=-6) — MR regression, needs deeper investigation
!74   OK          refactor/runtime-file-structure  pipeline running (include fix pushed last cycle)
!72   FIXED       agent-4-flux-perf-tuning        pixart/wan/z-image build_components missing fp8_scales kwarg
!68   WAITING     abstraction-inference-state      pipeline pending

Actions taken: 2 fixed, 1 diagnosed (needs follow-up), 3 waiting
```

Use these action labels:
- `OK` — pipeline green, up-to-date with master, nothing to do
- `WAITING` — pipeline running/pending, skipped this cycle
- `REBASED` — rebased onto master and pushed
- `FIXED` — diagnosed failure, fixed code, and pushed
- `DIAGNOSED` — diagnosed failure but could not fix (needs human review or deeper investigation)
- `SKIPPED` — pre-existing failure or infrastructure issue, not actionable

## Failure Diagnosis Reference

| Symptom | Likely cause | Where to look |
|---------|-------------|---------------|
| logit_cosine < 0.99 | Weight mapping wrong | `families/{family}.py:load_weights()` |
| NED = 1.0 (completely wrong text) | Model producing garbage | Build error, wrong runtime_strategy, broken graph |
| num_masks = 0 | Segmentation pipeline broken | C++ plugin or vision preprocessing |
| Build job failed | Compilation error | Job log, `src/` files |
| Unit test failed | Logic error | Job log, `tests/` files |
| OOM / timeout | Resource issue | Reduce `max_cache_length` in manifest, or skip |

## Important Rules

- **NEVER hardcode MR IIDs.** Always dynamically discover ALL open MRs via the API every cycle. MRs are created, closed, and un-drafted between cycles.
- ALL build/test commands go through `docker exec trtmc-dev-gb300-agent-1`
- Never force push to master
- Use `--force-with-lease` (not `--force`) when pushing rebases
- Do NOT modify unrelated code on other MR branches
- If a failure is clearly infrastructure/flaky (OOM, network timeout), note it and move on
- If stuck on the same MR for 3+ attempts, skip it and flag for human review
- Always use specific `git add` (not `git add -A`) to avoid committing secrets or unrelated files

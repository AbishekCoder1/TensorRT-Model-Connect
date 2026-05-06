# Transform Orchestrator + Dashboard Plan

## Goal

A Python-based orchestrator that launches multiple Claude Code CLI agents in
parallel to transform HF models to torch-trt. A web dashboard lets the user
monitor all agents, view side-by-side comparison outputs (text and images),
answer agent questions, and send feedback — all from a single browser tab.

---

## Architecture Overview

```
                    ┌──────────────────────────────┐
                    │     Browser (Dashboard UI)    │
                    │  - Agent cards with status     │
                    │  - Side-by-side comparisons    │
                    │  - Feedback input per agent    │
                    │  - Live log tails              │
                    └──────────────┬───────────────┘
                                   │ HTTP (localhost:8765)
                    ┌──────────────▼───────────────┐
                    │    dashboard.py (FastAPI)      │
                    │  - Serves UI                   │
                    │  - Polls agent status files     │
                    │  - Serves output artifacts      │
                    │  - POST /feedback → orchestrator│
                    └──────────────┬───────────────┘
                                   │ in-process
                    ┌──────────────▼───────────────┐
                    │   orchestrator.py              │
                    │  - Manages AgentSession objects │
                    │  - Launches claude --print      │
                    │  - Resumes sessions on feedback │
                    │  - Tracks session IDs           │
                    └───┬──────────┬──────────┬────┘
                        │          │          │
              ┌─────────▼──┐ ┌────▼─────┐ ┌──▼────────┐
              │  Worktree 1 │ │ Worktree 2│ │ Worktree 3│
              │  (llama3)   │ │ (phi4)    │ │ (pixart)  │
              │             │ │           │ │           │
              │ claude CLI  │ │ claude CLI│ │ claude CLI│
              │ session     │ │ session   │ │ session   │
              │             │ │           │ │           │
              │ .status.json│ │.status.json│.status.json│
              │ outputs/    │ │ outputs/  │ │ outputs/  │
              └─────────────┘ └───────────┘ └───────────┘
```

---

## Components

### 1. Orchestrator (`scripts/transform_orchestrator/orchestrator.py`)

Manages the lifecycle of Claude Code agent sessions.

**AgentSession class:**
```python
class AgentSession:
    agent_id: str           # e.g. "llama3-1b"
    hf_model: str           # e.g. "meta-llama/Llama-3.2-1B"
    branch: str             # e.g. "torchtrt/llama3"
    worktree: Path          # e.g. /home/.../worktrees/torchtrt-llama3
    session_id: str | None  # claude session ID for --resume
    container: str          # docker container name
    status: dict            # latest parsed .transform_status.json
    log_path: Path          # full agent log file
```

**Key methods:**
- `launch(initial_prompt)` — creates worktree, launches `claude --print` with
  `--output-format json`, parses session_id from response
- `resume(feedback)` — calls `claude --print --resume <session_id> -p <feedback>`
- `poll_status()` — reads `.transform_status.json` from worktree
- `get_log_tail(n=50)` — reads last N lines of agent log
- `get_outputs()` — lists files in `<worktree>/outputs/<model>/`

**Launch command:**
```python
["claude", "--print",
 "-p", prompt,
 "--output-format", "json",
 "--max-turns", "30",
 "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep"]
```

**Resume command:**
```python
["claude", "--print",
 "--resume", session_id,
 "-p", feedback,
 "--output-format", "json",
 "--max-turns", "30",
 "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep"]
```

**Worktree management:**
- Create: `git worktree add worktrees/<branch> -b <branch> master`
- Cleanup on completion: `git worktree remove worktrees/<branch>`
- Each worktree is a full repo clone sharing `.git` objects — lightweight

**Agent prompt injection:**
The orchestrator appends status-reporting instructions to the transform-model
skill prompt:

```
## Status Reporting (REQUIRED)

After completing each phase, write a status update:

    cat > .transform_status.json << 'STATUSEOF'
    {
      "model": "<hf_model>",
      "phase": <phase_number>,
      "phase_name": "<phase name>",
      "iteration": <iteration_count>,
      "status": "<running|awaiting_review|awaiting_feedback|done|failed>",
      "metrics": { ... },
      "outputs": {
        "hf_text": "...",
        "trt_text": "...",
        "images": ["outputs/<model>/hf_dog.png", "outputs/<model>/trt_dog.png"]
      },
      "question": "<question for user or null>",
      "error": "<last error message or null>"
    }
    STATUSEOF

When you set status to "awaiting_review" or "awaiting_feedback", STOP
and wait for the user to respond. Do not continue until you receive feedback.
```

### 2. Dashboard (`scripts/transform_orchestrator/dashboard.py`)

FastAPI web server that provides the monitoring UI.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard HTML page |
| GET | `/api/agents` | All agent statuses (JSON) |
| GET | `/api/agents/{id}` | Single agent status + outputs |
| GET | `/api/agents/{id}/log` | Last 100 lines of agent log |
| GET | `/api/agents/{id}/outputs/{filename}` | Serve output file (PNG, TXT, JSON) |
| POST | `/api/agents/{id}/feedback` | Send feedback to agent (resumes session) |
| POST | `/api/agents/{id}/approve` | Mark agent as approved, stop session |
| POST | `/api/agents/{id}/retry` | Resume agent with "try again" message |
| POST | `/api/launch` | Launch new agent (body: `{hf_model, branch, container}`) |
| GET | `/api/summary` | Aggregate status of all agents |

**Polling:** The frontend polls `/api/agents` every 5 seconds via JavaScript
fetch. No websockets needed — status files only change when an agent completes
a phase (every few minutes).

**Static file serving:** Output artifacts (PNGs, text files) are served directly
from each agent's worktree `outputs/` directory.

### 3. Frontend (`scripts/transform_orchestrator/templates/index.html`)

Single-page HTML with vanilla JavaScript (no framework needed).

**Agent card layout:**
Each agent gets a card showing:
- Model name, branch, container
- Current phase (progress bar: phases 0-7)
- Iteration count
- Status badge: RUNNING / AWAITING REVIEW / AWAITING FEEDBACK / DONE / FAILED
- Metrics table (when available): top1_match, cosine_sim, etc.
- Side-by-side comparison:
  - Text models: two text boxes (HF vs TRT output)
  - Diffusion models: two image columns (HF vs TRT for each prompt)
- Agent question (highlighted if present)
- Feedback input box + Send button
- Approve / Reject buttons
- Expandable live log tail (last 50 lines, auto-scrolls)
- Iteration history (collapsible): previous attempts with their metrics/errors

**Summary bar at top:**
- Total agents: N
- Running: X | Awaiting review: Y | Done: Z | Failed: W
- Overall time elapsed

### 4. Status File Contract (`.transform_status.json`)

Written by agent to worktree root. Read by dashboard.

```json
{
  "model": "meta-llama/Llama-3.2-1B",
  "phase": 6,
  "phase_name": "Report and Await Feedback",
  "iteration": 1,
  "status": "awaiting_review",
  "branch": "torchtrt/llama3",
  "started_at": "2026-03-31T14:00:00Z",
  "updated_at": "2026-03-31T14:12:34Z",
  "metrics": {
    "top1_match": 0.94,
    "cosine_sim": 0.998,
    "tokens_generated": 20
  },
  "outputs": {
    "hf_text": "The capital of France is Paris, a city known for...",
    "trt_text": "The capital of France is Paris, a city known for...",
    "images": [
      {"prompt": "a dog in snow", "hf": "outputs/llama3/hf_dog.png", "trt": "outputs/llama3/trt_dog.png"},
      {"prompt": "a red car", "hf": "outputs/llama3/hf_car.png", "trt": "outputs/llama3/trt_car.png"}
    ]
  },
  "question": null,
  "error": null,
  "history": [
    {"iteration": 1, "status": "failed", "error": "all-zero logits", "phase": 4}
  ]
}
```

**Status values:**
| Status | Meaning | Dashboard behavior |
|--------|---------|-------------------|
| `running` | Agent actively working | Show spinner + log tail |
| `awaiting_review` | Comparison output ready | Show comparisons + approve/reject buttons |
| `awaiting_feedback` | Agent has a question | Highlight question + show feedback input |
| `done` | User approved | Green badge, no actions |
| `failed` | Agent gave up (max iterations) | Red badge, show error |

---

## File Structure

```
scripts/transform_orchestrator/
  __init__.py
  orchestrator.py          # AgentSession class, launch/resume/poll logic
  dashboard.py             # FastAPI app, API endpoints, serves UI
  templates/
    index.html             # Dashboard UI (single page, vanilla JS)
  static/
    style.css              # Dashboard styling
  cli.py                   # CLI entry point: parse args, start dashboard + orchestrator

worktrees/                 # Git worktrees (gitignored, already in .gitignore)
  torchtrt-llama3/
    .transform_status.json
    outputs/llama3-1b/
      hf_output.txt
      trt_output.txt
      diff_logits.json
  torchtrt-phi4/
    ...
```

---

## CLI Interface

```bash
# Launch dashboard + agents for specific models
python3 scripts/transform_orchestrator/cli.py \
  --models "meta-llama/Llama-3.2-1B" "microsoft/phi-4-mini" "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS" \
  --container trtmc-dev-torchtrt \
  --port 8765

# Launch dashboard only (agents launched via UI)
python3 scripts/transform_orchestrator/cli.py --port 8765

# Add a model to running dashboard
curl -X POST http://localhost:8765/api/launch \
  -d '{"hf_model": "Qwen/Qwen3-0.6B", "branch": "torchtrt/qwen3", "container": "trtmc-dev-torchtrt"}'
```

**What happens on launch:**
1. Creates git worktree for each model
2. Starts FastAPI server on `--port`
3. For each model: builds prompt from `skills/transform-model/SKILL.md` +
   status reporting instructions, launches `claude --print`
4. Opens browser to `http://localhost:8765`
5. Polls `.transform_status.json` in each worktree every 5s
6. When user sends feedback via UI, resumes the agent's claude session

---

## Dependencies

- `fastapi` — web framework (lightweight, async)
- `uvicorn` — ASGI server for FastAPI
- No frontend framework — vanilla HTML/JS/CSS

Install: `pip install fastapi uvicorn`

These are orchestrator-only dependencies, not added to the build packages.

---

## Implementation Order

### Phase 1: Core orchestrator (no UI)

1. `orchestrator.py` — AgentSession with launch/resume/poll
2. `cli.py` — CLI that launches agents and prints status to terminal
3. Test with one model end-to-end: launch agent, poll status, send feedback
   via CLI, verify agent resumes

**Verification:** Launch one agent for Qwen3-0.6B, watch it build a bundle,
see `.transform_status.json` appear, send feedback via `--resume`, confirm
agent continues.

### Phase 2: Dashboard backend

4. `dashboard.py` — FastAPI app with all API endpoints
5. Wire orchestrator into dashboard (in-process, shared state)
6. Test API endpoints with curl

**Verification:** `curl localhost:8765/api/agents` returns agent statuses.
`curl -X POST localhost:8765/api/agents/llama3/feedback -d '{"message": "looks good"}'`
resumes agent.

### Phase 3: Dashboard frontend

7. `templates/index.html` — agent cards, polling, feedback forms
8. `static/style.css` — clean layout
9. Image serving for diffusion model comparisons

**Verification:** Open browser, see all agent cards, send feedback through UI,
see agent resume and update status.

### Phase 4: Polish

10. Auto-open browser on launch
11. Iteration history in agent cards
12. Log viewer (expandable per agent)
13. Summary bar with aggregate stats
14. Error handling for crashed agents / lost sessions

---

## Open Questions

1. **Worktree vs workspace?** Git worktrees are lightweight but share the same
   `.git`. If agents modify the same file (e.g., CLAUDE.md, KNOWN_ISSUES.md),
   there could be conflicts on merge. Alternative: use `bootstrap_workspace.sh`
   for full isolation like autopilot does. Tradeoff: heavier but safer.

2. **Container sharing?** All agents currently target one container
   (`trtmc-dev-torchtrt`). If two agents try to build bundles simultaneously,
   GPU memory may be an issue. Options:
   - Sequential GPU access (orchestrator queues build phases)
   - Multiple containers (one per agent, like autopilot's agent-1..agent-4)
   - Let agents handle OOM and retry

3. **Session persistence across dashboard restarts?** If the dashboard process
   dies, session IDs are lost. Could persist session IDs to a JSON file so the
   dashboard can reconnect to running agents on restart.

4. **Max concurrent agents?** GPU memory is the bottleneck during build phases.
   Text models need ~4-8 GB, diffusion models need ~16-24 GB. On a single GPU,
   2-3 concurrent builds may be the practical limit. The orchestrator could
   implement a build queue while allowing multiple agents to do non-GPU work
   (reading docs, writing code) in parallel.

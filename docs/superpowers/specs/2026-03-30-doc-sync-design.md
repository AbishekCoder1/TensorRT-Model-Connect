# Doc-Sync: Automated Documentation Maintenance Skill

**Date:** 2026-03-30
**Status:** Proposed
**Approach:** Multi-phase skill with focused MRs (Approach 2)

## Problem

Design context gets lost as the repo evolves. `docs/WORKLOG.md` (138KB) served
as an engineering log but is no longer actively maintained. Wiki pages in
`docs/wiki/` contain factual claims (file paths, strategy counts, family counts,
test tables) that drift as code changes faster than documentation. The
traceability matrix (`ARCH-*/UD-*/UT-*/IT-*`) has gaps as new tests and source
files are added without trace ID annotations.

Two complementary problems:
1. **Context capture** — design decisions should be recorded as ADRs at
   MR-submit time, not reconstructed retrospectively from diffs.
2. **Documentation maintenance** — existing docs (ADRs, wiki pages, traceability
   matrix) must be kept in sync with the codebase by a daily automated scan.

## Solution Overview

Two skill modifications:

1. **`submit-gitlab-mr` enhancement** — draft ADRs at MR submission time when
   high-confidence design decision triggers are detected. The ADR is committed
   to the feature branch and reviewed as part of the MR.

2. **`doc-sync` skill** — daily scan that maintains existing documentation via
   three phases, each producing its own MR for independent review.

## ADR System

### Location and Format

ADRs live in `docs/context/adr/` using MADR format with sequential numbering:

```
docs/context/adr/
  README.md              # Index table (number, title, status, date)
  0001-<slug>.md
  0002-<slug>.md
  ...
```

ADR template:

```markdown
---
number: NNNN
title: <decision title>
status: Proposed | Accepted | Deprecated | Superseded
date: YYYY-MM-DD
source_commits: [sha1, sha2, ...]
superseded_by: NNNN  # optional
---

## Context

<What problem or situation prompted this decision?>

## Decision

<What was decided?>

## Considered Alternatives

<Rejected approaches and why. If not captured: "Not captured — review and add
if known.">

## Consequences

<Implications: new constraints, capabilities, maintenance burden, performance.>
```

### ADR Creation (submit-gitlab-mr Enhancement)

When submitting an MR, the skill analyzes the diff (`git diff master...HEAD`)
for high-confidence triggers:

| Signal | Detection method |
|--------|-----------------|
| New runtime strategy | New `REGISTER_PIPELINE_PLUGIN` call in `src/runtime/plugins/` |
| New family plugin | New `.py` in `tensorrt_model_connect/tensorrt_model_connect/families/` with `plugin` attribute |
| New pipeline class | New `.cpp/.h` pair in `src/runtime/pipelines/` |
| Config schema change | New fields parsed in `pipeline_plugin.cpp` or `config.py` |
| New E2E task strategy | New runner in `tests/e2e_harness/runners/` |
| New comparator/reference | New file in `tests/e2e_harness/comparators/` or `references/` |
| Architectural refactor | Large-scale file moves/renames across `src/runtime/` |

If triggers fire:
1. Draft ADR(s) into `docs/context/adr/NNNN-<slug>.md`
2. Update `docs/context/adr/README.md` index
3. Commit to the feature branch
4. Proceed with MR creation as normal

The ADR has full conversation context — richer than a retrospective reconstruction.

**Opt-out:** `--no-adr` flag skips ADR generation for trivial MRs.

**Numbering:** Read highest existing number, increment. Conflicts from
simultaneous MRs resolved at merge time (second merger renumbers).

**What does NOT get an ADR:** Bug fixes, new tests without architectural impact,
dependency updates, documentation-only changes, new E2E manifests for existing
families.

## Doc-Sync Skill

### Identity and Invocation

**Skill name:** `doc-sync`
**Location:** `.claude/skills/doc-sync/SKILL.md`

Invocation modes:
- Interactive: `/loop 24h /doc-sync`
- Cron: `cd /workspace/users/yifeif/workspaces/<agent-id>/tensorrt-model-connect && claude --print -p "/doc-sync"`
- Manual: `/doc-sync`

### State Management

**State file:** `docs/context/.last_scan_sha`

- Contains the commit SHA of the last completed scan
- Skill diffs from this SHA to HEAD
- First run (no state file): uses last 7 days of commits as bootstrap
- Updated only after all phases complete successfully
- If a phase fails (non-auth), the SHA is still advanced after remaining phases

### Branch Naming

Each phase creates: `doc-sync/<phase>-<YYYY-MM-DD>`
- `doc-sync/context-2026-03-30`
- `doc-sync/drift-2026-03-30`
- `doc-sync/traceability-2026-03-30`

MR labels: `doc-sync` + phase name (`context`, `drift`, `traceability`).

### Idempotency

Before creating a branch, check if `doc-sync/<phase>-<date>` already exists
on remote. If so, skip that phase. Running twice on the same range produces
the same result.

### Phase 1 — ADR Maintenance

Keeps existing ADRs accurate and current. Does NOT create new ADRs (that is
`submit-gitlab-mr`'s job).

| Check | Detection | Action |
|-------|-----------|--------|
| Superseded decisions | ADR references a pattern/file that's been replaced | Update status to `Superseded`, link to superseding commit/ADR |
| Dead references | ADR mentions files, classes, or strategies that no longer exist | Fix references to current names/paths |
| Stale consequences | ADR claims a constraint/limitation that's no longer true | Fix factual claims inline |
| Status lifecycle | ADRs in `Proposed` status >30 days with decision clearly implemented | Promote to `Accepted` |
| Orphaned ADRs | ADR's `source_commits` on a reverted/unmerged branch | Mark as `Deprecated` with explanation |
| Index drift | `README.md` out of sync with actual ADR files | Rebuild index |

### Phase 2 — Wiki Drift Repair

Verifies factual claims in `docs/wiki/*.md` against the codebase and fixes
everything. The MR review is the human gate.

| Claim type | Source of truth |
|------------|----------------|
| File paths | Glob against repo |
| Strategy counts | Scan `REGISTER_PIPELINE_PLUGIN` calls |
| Family plugin counts | Count `.py` files in `families/` with `plugin` attribute |
| E2E model counts | Count `.json` files in `tests/e2e/models/` |
| LOC claims | `wc -l` on referenced files |
| Class/function names | Grep for referenced symbols |
| Strategy-to-plugin tables | Cross-reference registry with documented tables |
| Test file tables | Cross-reference `tests/` with documented test tables |
| Source layout tree | Diff actual directory tree against Source-Layout.md |
| Behavioral descriptions | Read referenced source code, rewrite prose to match |

Wiki pages scanned (priority order):
1. `Architecture-Overview.md` — strategy tables, plugin counts
2. `Source-Layout.md` — directory tree, file descriptions
3. `Testing-and-Validation.md` — test file tables, counts, layer descriptions
4. `Traceability-Matrix.md` — trace ID references
5. `Static-Design.md` — class names, interface names
6. `Pipeline-Deep-Dive.md` — strategy dispatch flow
7. All remaining wiki pages — dead file/symbol references

**Fix policy:** Auto-fix everything. No "flag only" category.

MR description format:

```markdown
## Wiki drift fixes (auto-generated by doc-sync)

### Changes
- Architecture-Overview.md: strategy count 17 → 19, added t5 and object_detection rows
- Source-Layout.md: added `src/runtime/plugins/t5_plugin.cpp`, updated directory tree
- Testing-and-Validation.md: test file count 61 → 63, updated layer table
- Pipeline-Deep-Dive.md: rewrote legacy normalization section to match current logic
```

### Phase 3 — Traceability Audit

Purely mechanical scan of the ARCH-*/UD-*/UT-*/IT-* trace ID system.

| Check | Detection | Action |
|-------|-----------|--------|
| Tests missing trace IDs | Scan test docstrings/comments for missing `Trace:` | Add trace ID annotations |
| Orphaned trace IDs | UT-*/IT-* in matrix referencing deleted test files | Remove entries |
| Unverified ARCH-* | ARCH-* entries with zero linked UT-*/IT-* | Add links to matching tests |
| Missing UD-* coverage | Source files in `src/` and `tensorrt_model_connect/` not referenced by any UD-* | Add UD-* entries |
| Stale UD-* references | UD-* references nonexistent source file | Fix paths or remove |
| Unlinked E2E manifests | Manifests in `tests/e2e/models/` with no IT-* entry | Add IT-* entries |
| Trace ID format violations | Malformed IDs in test docstrings | Fix formatting |

Coverage metrics included in MR description:

```markdown
## Traceability audit (auto-generated by doc-sync)

### Coverage
- ARCH-* entries: 12/12 have linked tests (100%)
- UD-* coverage: 87/94 source files traced (92.6%)
- Test trace IDs: 203/215 test functions annotated (94.4%)
- E2E manifests linked: 72/72 (100%)

### Changes
- Added 4 IT-* entries for new E2E manifests
- Added 7 UD-* entries for new source files
- Removed 2 orphaned UT-* references
- Fixed 3 stale file paths in UD-* entries
```

## Execution Flow

```
1. Read .last_scan_sha (or use 7-day bootstrap)
2. git fetch origin && git log <last_sha>..HEAD → change set
3. If change set is empty → update .last_scan_sha, exit

Phase 1 — ADR Maintenance:
4. Scan existing ADRs against codebase
5. If fixes needed → create branch, commit, open MR

Phase 2 — Wiki Drift:
6. Run all wiki checks against codebase
7. If fixes needed → create branch, commit, open MR

Phase 3 — Traceability:
8. Run full trace ID scan
9. If fixes needed → create branch, commit, open MR

10. Update .last_scan_sha to HEAD
11. Write summary report to stdout
```

## Error Handling

| Failure | Behavior |
|---------|----------|
| `glab` auth fails | Abort, report error, don't advance SHA |
| Branch push fails (permissions) | Abort that phase, continue to next phase |
| Wiki page has unexpected format | Parse best-effort and fix what's fixable. No skipping. |
| No changes in any phase | Advance SHA, log "nothing to do" |
| Large change set | Process all of it. No caps. |

The only reasons to stop are authorization failures. Everything else gets
processed.

## Deliverables

### New files

| File | Purpose |
|------|---------|
| `.claude/skills/doc-sync/SKILL.md` | Daily doc-sync skill (Phases 1-3) |
| `docs/context/adr/README.md` | ADR index table |
| `docs/context/.last_scan_sha` | SHA marker for incremental scanning |

### Modified files

| File | Change |
|------|--------|
| `~/.claude/skills/submit-gitlab-mr/SKILL.md` | Enhanced with ADR creation logic |

## Out of Scope

- CLAUDE.md changes
- Creating new wiki pages (maintains existing ones only)
- Writing test implementations (reports traceability gaps, adds trace IDs)

#!/usr/bin/env python3
"""Autonomous, conservative merge loop for GitLab MRs targeting ai-staging."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    project: str
    remote: str
    target: str
    source_prefixes: tuple[str, ...]
    required_labels: tuple[str, ...]
    dry_run: bool
    max_actions: int
    poll_seconds: int
    close_empty: bool
    work_branch_prefix: str


def log(message: str) -> None:
    print(message, flush=True)


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log("+ " + shlex.join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        env={**os.environ, **(env or {})},
    )


def git(args: list[str], *, check: bool = True, dry_run: bool = False) -> str:
    result = run(["git", *args], check=check, capture=True, dry_run=dry_run)
    return result.stdout.strip()


def api(cfg: Config, path: str, *, method: str | None = None, fields: dict[str, str] | None = None) -> Any:
    cmd = ["glab", "api"]
    if method:
        cmd.extend(["-X", method])
    cmd.append(path)
    for key, value in (fields or {}).items():
        cmd.extend(["-f", f"{key}={value}"])
    result = run(cmd, dry_run=cfg.dry_run and bool(method or fields))
    if cfg.dry_run and bool(method or fields):
        return None
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def encoded_project(project: str) -> str:
    return project if project.isdecimal() else urllib.parse.quote(project, safe="")


def infer_project(remote: str) -> str:
    remote_url = git(["remote", "get-url", remote])
    if "://" in remote_url:
        path = urllib.parse.urlparse(remote_url).path.lstrip("/")
    elif "@" in remote_url and ":" in remote_url:
        path = remote_url.split(":", 1)[1]
    else:
        path = remote_url
    if path.endswith(".git"):
        path = path[:-4]
    if not path or "/" not in path:
        raise SystemExit(f"Cannot infer GitLab project from remote URL: {remote_url!r}")
    return path


def assert_clean_worktree() -> None:
    status = git(["status", "--porcelain"])
    if status:
        raise SystemExit("Refusing to run with a dirty worktree:\n" + status)


def fetch_branch(remote: str, branch: str) -> None:
    git(["fetch", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}"])


def list_candidate_mrs(cfg: Config) -> list[dict[str, Any]]:
    mrs: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {
                "state": "opened",
                "target_branch": cfg.target,
                "per_page": "100",
                "page": str(page),
                "order_by": "created_at",
                "sort": "asc",
            }
        )
        batch = api(cfg, f"/projects/{cfg.project}/merge_requests?{query}")
        if not isinstance(batch, list):
            raise SystemExit(f"Unexpected MR list response: {batch!r}")
        if not batch:
            break
        for mr in batch:
            source = str(mr.get("source_branch") or "")
            if source.startswith(cfg.source_prefixes):
                mrs.append(mr)
        if len(batch) < 100:
            break
        page += 1
    return mrs


def mr_details(cfg: Config, iid: int) -> dict[str, Any]:
    details = api(cfg, f"/projects/{cfg.project}/merge_requests/{iid}?include_rebase_in_progress=true")
    if not isinstance(details, dict):
        raise SystemExit(f"Unexpected MR details response for !{iid}: {details!r}")
    return details


def approvals_left(cfg: Config, iid: int) -> int | None:
    result = api(cfg, f"/projects/{cfg.project}/merge_requests/{iid}/approvals")
    if not isinstance(result, dict):
        return None
    value = result.get("approvals_left")
    return int(value) if value is not None else None


def pipeline_for_mr(cfg: Config, mr: dict[str, Any]) -> dict[str, Any] | None:
    pipeline = mr.get("head_pipeline")
    if not isinstance(pipeline, dict) or not pipeline.get("id"):
        return None
    full = api(cfg, f"/projects/{cfg.project}/pipelines/{pipeline['id']}")
    return full if isinstance(full, dict) else pipeline


def print_queue(cfg: Config, mrs: list[dict[str, Any]]) -> None:
    log(f"Open MRs targeting {cfg.target}: {len(mrs)}")
    log("IID   status          pipeline   approvals  source")
    log("----  --------------  ---------  ---------  --------------------------------")
    for mr in mrs:
        iid = int(mr["iid"])
        details = mr_details(cfg, iid)
        pipeline = details.get("head_pipeline") or {}
        left = approvals_left(cfg, iid)
        log(
            f"!{iid:<3}  "
            f"{str(details.get('detailed_merge_status') or '-'):<14}  "
            f"{str(pipeline.get('status') or '-'):<9}  "
            f"{str(left if left is not None else '-'):<9}  "
            f"{details.get('source_branch')}"
        )


def eligible(cfg: Config, mr: dict[str, Any]) -> tuple[bool, str]:
    if mr.get("draft") or mr.get("work_in_progress"):
        return False, "draft"
    if mr.get("source_project_id") != mr.get("target_project_id"):
        return False, "source project differs from target project"
    if mr.get("rebase_in_progress"):
        return False, "rebase in progress"
    labels = set(mr.get("labels") or [])
    missing_labels = [label for label in cfg.required_labels if label not in labels]
    if missing_labels:
        return False, "missing labels: " + ",".join(missing_labels)

    left = approvals_left(cfg, int(mr["iid"]))
    if left not in (None, 0):
        return False, f"approvals_left={left}"

    pipeline = pipeline_for_mr(cfg, mr)
    if not pipeline:
        return False, "no head pipeline"
    if pipeline.get("status") != "success":
        return False, f"pipeline={pipeline.get('status')}"
    pipeline_sha = pipeline.get("sha")
    mr_sha = mr.get("sha")
    if pipeline_sha and mr_sha and pipeline_sha != mr_sha:
        return False, f"pipeline SHA {pipeline_sha} != MR SHA {mr_sha}"

    return True, "eligible"


def rebase_in_progress() -> bool:
    merge_path = Path(git(["rev-parse", "--git-path", "rebase-merge"]))
    apply_path = Path(git(["rev-parse", "--git-path", "rebase-apply"]))
    return merge_path.exists() or apply_path.exists()


def unmerged_paths() -> list[str]:
    result = run(["git", "diff", "--name-only", "--diff-filter=U", "-z"], check=False)
    if result.returncode != 0:
        raise SystemExit(result.stderr)
    return [path for path in result.stdout.split("\0") if path]


def resolve_conflicts_target_wins() -> int:
    paths = unmerged_paths()
    if not paths:
        return 0

    resolved = 0
    for path in paths:
        checkout = run(["git", "checkout", "--ours", "--", path], check=False)
        if checkout.returncode == 0:
            run(["git", "add", "--", path], check=False)
        else:
            run(["git", "rm", "-f", "--ignore-unmatch", "--", path], check=False)
        resolved += 1
    return resolved


def continue_rebase_target_wins(max_rounds: int = 50) -> bool:
    had_conflict = False
    for _ in range(max_rounds):
        if not rebase_in_progress():
            return had_conflict

        resolved = resolve_conflicts_target_wins()
        if resolved:
            had_conflict = True

        cont = run(["git", "rebase", "--continue"], check=False, env={"GIT_EDITOR": "true"})
        if cont.returncode == 0:
            continue

        combined = (cont.stdout or "") + (cont.stderr or "")
        if "No changes" in combined or "previous cherry-pick is now empty" in combined:
            skip = run(["git", "rebase", "--skip"], check=False)
            if skip.returncode == 0:
                continue

        if unmerged_paths():
            continue
        raise SystemExit("Could not continue rebase:\n" + combined)

    raise SystemExit("Rebase conflict resolution exceeded iteration limit")


def branch_is_empty_against_target(cfg: Config) -> bool:
    result = run(["git", "diff", "--quiet", f"{cfg.remote}/{cfg.target}", "HEAD"], check=False)
    return result.returncode == 0


def push_source(cfg: Config, source_branch: str, *, skip_ci: bool, expected_sha: str) -> str:
    sha = git(["rev-parse", "HEAD"])
    cmd = ["git", "push"]
    if skip_ci:
        cmd.extend(["-o", "ci.skip"])
    cmd.append(f"--force-with-lease=refs/heads/{source_branch}:{expected_sha}")
    cmd.extend([cfg.remote, f"HEAD:refs/heads/{source_branch}"])
    run(cmd, dry_run=cfg.dry_run)
    return sha


def wait_for_mergeable_sha(cfg: Config, iid: int, sha: str) -> dict[str, Any] | None:
    for _ in range(24):
        mr = mr_details(cfg, iid)
        status = mr.get("detailed_merge_status")
        if mr.get("sha") == sha and status == "mergeable":
            return mr
        if status in {"cannot_be_merged", "conflict", "not_open", "draft_status"}:
            return None
        time.sleep(cfg.poll_seconds)
    return None


def merge_mr(cfg: Config, iid: int, sha: str) -> bool:
    fields = {"sha": sha, "should_remove_source_branch": "false"}
    result = api(cfg, f"/projects/{cfg.project}/merge_requests/{iid}/merge", method="PUT", fields=fields)
    if cfg.dry_run:
        return True
    if not isinstance(result, dict):
        log(f"!{iid}: unexpected merge response: {result!r}")
        return False
    log(f"!{iid}: merged at {result.get('merged_at') or result.get('merge_commit_sha') or 'unknown'}")
    return True


def close_empty_mr(cfg: Config, iid: int) -> None:
    if cfg.dry_run:
        log(f"!{iid}: DRY RUN would close because it is empty after rebasing onto {cfg.target}")
        return
    fields = {"state_event": "close"}
    api(cfg, f"/projects/{cfg.project}/merge_requests/{iid}", method="PUT", fields=fields)
    log(f"!{iid}: closed because it is empty after rebasing onto {cfg.target}")


def process_mr(cfg: Config, mr: dict[str, Any]) -> str:
    iid = int(mr["iid"])
    source = str(mr["source_branch"])
    fetch_branch(cfg.remote, cfg.target)
    fetch_branch(cfg.remote, source)
    source_remote_sha = git(["rev-parse", f"{cfg.remote}/{source}"])

    work_branch = f"{cfg.work_branch_prefix}{iid}"
    git(["switch", "-C", work_branch, f"{cfg.remote}/{source}"])

    rebase = run(["git", "rebase", "--empty=drop", f"{cfg.remote}/{cfg.target}"], check=False)
    had_conflict = False
    if rebase.returncode != 0:
        had_conflict = continue_rebase_target_wins()

    if branch_is_empty_against_target(cfg):
        if cfg.close_empty:
            close_empty_mr(cfg, iid)
            return "closed-empty"
        log(f"!{iid}: empty after rebase; leaving open for review")
        return "empty"

    if had_conflict:
        sha = push_source(cfg, source, skip_ci=False, expected_sha=source_remote_sha)
        if cfg.dry_run:
            log(f"!{iid}: DRY RUN would push target-wins conflict resolution at {sha} without ci.skip")
            return "dry-run-conflict-resolution"
        log(f"!{iid}: pushed target-wins conflict resolution at {sha}; waiting for new CI")
        return "pushed-conflict-resolution"

    sha = push_source(cfg, source, skip_ci=True, expected_sha=source_remote_sha)
    if cfg.dry_run:
        log(f"!{iid}: DRY RUN would push clean rebase at {sha} with ci.skip, then merge by exact SHA")
        return "dry-run-clean-rebase"

    refreshed = wait_for_mergeable_sha(cfg, iid, sha)
    if not refreshed:
        log(f"!{iid}: pushed clean rebase at {sha}, but GitLab did not report mergeable; retry later")
        return "pushed-clean-rebase"

    return "merged" if merge_mr(cfg, iid, sha) else "merge-rejected"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("CI_PROJECT_PATH"), help="GitLab project path or numeric id")
    parser.add_argument("--remote", default=os.environ.get("AI_STAGING_REMOTE", "origin"))
    parser.add_argument("--target", default=os.environ.get("AI_STAGING_BRANCH", "ai-staging"))
    parser.add_argument("--source-prefix", action="append", default=None, help="source branch prefix; repeatable")
    parser.add_argument(
        "--required-label",
        action="append",
        default=None,
        help="label required before an MR can be merged; repeatable. Default: ai:staging-mr",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-actions", type=int, default=1, help="number of merge/push actions to perform this run")
    parser.add_argument("--once", action="store_true", help="alias for --max-actions 1")
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--close-empty", action="store_true", help="close MRs that become empty after rebase")
    parser.add_argument("--work-branch-prefix", default="autopilot/mr-")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = args.project or infer_project(args.remote)
    cfg = Config(
        project=encoded_project(project),
        remote=args.remote,
        target=args.target,
        source_prefixes=tuple(args.source_prefix or ["agent-2-"]),
        required_labels=tuple(args.required_label or ["ai:staging-mr"]),
        dry_run=args.dry_run,
        max_actions=1 if args.once else args.max_actions,
        poll_seconds=args.poll_seconds,
        close_empty=args.close_empty,
        work_branch_prefix=args.work_branch_prefix,
    )

    assert_clean_worktree()
    mrs = list_candidate_mrs(cfg)
    print_queue(cfg, mrs)

    actions = 0
    for candidate in mrs:
        if actions >= cfg.max_actions:
            break

        details = mr_details(cfg, int(candidate["iid"]))
        ok, reason = eligible(cfg, details)
        if not ok:
            log(f"!{details['iid']}: skip: {reason}")
            continue

        result = process_mr(cfg, details)
        log(f"!{details['iid']}: result={result}")
        actions += 1

    if actions == 0:
        log("No eligible MR processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

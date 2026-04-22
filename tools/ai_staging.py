#!/usr/bin/env python3
"""Maintain the AI staging branch and route AI-generated merge requests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_BRANCH = "ai-staging"
DEFAULT_REMOTE = "origin"
DEFAULT_SOURCE_PREFIXES = ("ai-task-",)
DEFAULT_PROMOTION_PREFIX = "ai-staging-promotion"
AI_STAGING_MR_LABELS = "ai-generated,ai:staging-mr"
AI_PROMOTION_MR_LABELS = "ai:promotion"
ACTIVE_PIPELINE_STATUSES = {"created", "waiting_for_resource", "preparing", "pending", "running"}
FAILED_PIPELINE_STATUSES = {"failed", "canceled"}


@dataclass(frozen=True)
class Config:
    remote: str
    branch: str
    project: str
    source_prefixes: tuple[str, ...]
    dry_run: bool
    require_api_token: bool


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    dry_run: bool = False,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    location = f"(cd {shlex.quote(cwd)} && " if cwd else ""
    suffix = ")" if cwd else ""
    print("+ " + location + shlex.join(cmd) + suffix, file=sys.stderr)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    result = subprocess.run(cmd, check=False, capture_output=capture, text=True, cwd=cwd)
    if check and result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        message = f"Command failed with exit code {result.returncode}: {shlex.join(cmd)}"
        if details:
            message += "\n" + details
        raise SystemExit(message)
    return result


def git(args: list[str], *, check: bool = True, dry_run: bool = False, cwd: str | None = None) -> str:
    result = run(["git", *args], check=check, capture=True, dry_run=dry_run, cwd=cwd)
    return result.stdout.strip()


def infer_project_path(remote: str) -> str:
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
        raise SystemExit(
            f"Could not infer GitLab project path from remote URL: {remote_url!r}. "
            "Pass --project explicitly."
        )
    return path


def encoded_project_path(project: str) -> str:
    return project if project.isdecimal() else urllib.parse.quote(project, safe="")


def config_from_args(args: argparse.Namespace) -> Config:
    project = args.project or os.environ.get("CI_PROJECT_PATH") or infer_project_path(args.remote)
    prefixes = tuple(args.source_prefix or DEFAULT_SOURCE_PREFIXES)
    return Config(
        remote=args.remote,
        branch=args.branch,
        project=encoded_project_path(project),
        source_prefixes=prefixes,
        dry_run=args.dry_run,
        require_api_token=getattr(args, "require_api_token", False),
    )


def fetch_branch(cfg: Config, branch: str) -> None:
    git(
        [
            "fetch",
            cfg.remote,
            f"+refs/heads/{branch}:refs/remotes/{cfg.remote}/{branch}",
        ]
    )


def remote_ref(cfg: Config, branch: str) -> str:
    return f"{cfg.remote}/{branch}"


def remote_branch_exists(cfg: Config, branch: str) -> bool:
    output = git(["ls-remote", "--heads", cfg.remote, branch])
    return bool(output)


def local_branch_exists(branch: str) -> bool:
    result = run(["git", "rev-parse", "--verify", f"refs/heads/{branch}"], check=False)
    return result.returncode == 0


def assert_clean_worktree() -> None:
    status = git(["status", "--porcelain"])
    if status:
        raise SystemExit(
            "Refusing to operate with local changes present. Commit, stash, or use a clean worktree.\n"
            + status
        )


def ensure_branch(cfg: Config, *, source_branch: str = "master") -> bool:
    fetch_branch(cfg, source_branch)
    if remote_branch_exists(cfg, cfg.branch):
        print(f"{cfg.remote} branch exists: {cfg.branch}")
        fetch_branch(cfg, cfg.branch)
        return False

    source = f"refs/remotes/{cfg.remote}/{source_branch}:refs/heads/{cfg.branch}"
    run(["git", "push", cfg.remote, source], dry_run=cfg.dry_run)
    print(f"created {cfg.remote}/{cfg.branch} from {cfg.remote}/{source_branch}")
    return True


def sync_branch(cfg: Config, *, push: bool) -> None:
    assert_clean_worktree()
    ensure_branch(cfg)

    if cfg.dry_run:
        print(f"would sync {cfg.remote}/{cfg.branch} with {cfg.remote}/master")
        if push:
            print(f"would push synced HEAD to {cfg.remote}/{cfg.branch}")
        return

    worktree_path = tempfile.mkdtemp(prefix=f"{cfg.branch}-sync-")
    keep_worktree = False
    contains_master = run(
        ["git", "worktree", "add", "--detach", worktree_path, f"{cfg.remote}/{cfg.branch}"],
        check=False,
    )
    if contains_master.returncode != 0:
        shutil.rmtree(worktree_path, ignore_errors=True)
        details = (contains_master.stderr or contains_master.stdout or "").strip()
        raise SystemExit(f"Could not create temporary sync worktree at {worktree_path}.\n{details}")

    try:
        contains_master = run(
            ["git", "merge-base", "--is-ancestor", f"{cfg.remote}/master", "HEAD"],
            check=False,
            cwd=worktree_path,
        )
        if contains_master.returncode == 0:
            print(f"{cfg.branch} already contains {cfg.remote}/master")
        else:
            git(["merge", "--no-edit", f"{cfg.remote}/master"], cwd=worktree_path)

        if push:
            run(["git", "push", cfg.remote, f"HEAD:refs/heads/{cfg.branch}"], cwd=worktree_path)
    except BaseException:
        keep_worktree = True
        print(f"Preserving failed sync worktree for inspection: {worktree_path}", file=sys.stderr)
        raise
    finally:
        if not keep_worktree:
            run(["git", "worktree", "remove", "--force", worktree_path], check=False)
            shutil.rmtree(worktree_path, ignore_errors=True)


def infer_gitlab_server_url(remote: str) -> str:
    remote_url = git(["remote", "get-url", remote])
    if "://" in remote_url:
        parsed = urllib.parse.urlparse(remote_url)
        if parsed.hostname:
            return f"https://{parsed.hostname}"
    if "@" in remote_url and ":" in remote_url:
        host = remote_url.split("@", 1)[1].split(":", 1)[0]
        if host:
            return f"https://{host}"
    raise SystemExit(
        f"Could not infer GitLab server URL from remote URL: {remote_url!r}. "
        "Set CI_API_V4_URL."
    )


def gitlab_api_base_url(cfg: Config) -> str:
    if os.environ.get("CI_API_V4_URL"):
        return os.environ["CI_API_V4_URL"].rstrip("/")
    return infer_gitlab_server_url(cfg.remote).rstrip("/") + "/api/v4"


def gitlab_token_header() -> tuple[str, str] | None:
    token = os.environ.get("AI_STAGING_BOT_TOKEN")
    if token:
        return "PRIVATE-TOKEN", token
    return None


def missing_token_message() -> str:
    return (
        "AI_STAGING_BOT_TOKEN is required for GitLab REST API access. "
        "Create a masked CI/CD variable with a project access token that has API scope."
    )


def glab_api_json(path: str, *, method: str | None = None, fields: dict[str, str] | None = None) -> Any:
    cmd = ["glab", "api"]
    if method:
        cmd.extend(["-X", method])
    cmd.append(path)
    for key, value in (fields or {}).items():
        cmd.extend(["-f", f"{key}={value}"])
    result = run(cmd)
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def http_api_json(
    cfg: Config,
    path: str,
    *,
    method: str | None = None,
    fields: dict[str, str] | None = None,
) -> Any:
    token_header = gitlab_token_header()
    if not token_header:
        raise SystemExit(missing_token_message())

    data = urllib.parse.urlencode(fields or {}).encode() if fields else None
    request_method = method or ("POST" if data else "GET")
    url = gitlab_api_base_url(cfg) + path
    print(f"+ {request_method} {url}", file=sys.stderr)
    request = urllib.request.Request(url, data=data, method=request_method)
    request.add_header(*token_header)
    if data:
        request.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(request) as response:
            payload = response.read().decode()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise SystemExit(f"GitLab API failed: HTTP {exc.code} {exc.reason}: {body}") from exc

    if not payload.strip():
        return None
    return json.loads(payload)


def gitlab_api_json(
    cfg: Config,
    path: str,
    *,
    method: str | None = None,
    fields: dict[str, str] | None = None,
) -> Any:
    is_write = (method or "").upper() in {"POST", "PUT", "PATCH", "DELETE"} or bool(fields)
    if cfg.dry_run and is_write:
        print(f"+ DRY RUN GitLab API {method or 'POST'} {path} {fields or {}}", file=sys.stderr)
        return None
    if gitlab_token_header():
        return http_api_json(cfg, path, method=method, fields=fields)
    if cfg.require_api_token:
        raise SystemExit(missing_token_message())
    return glab_api_json(path, method=method, fields=fields)


def gitlab_paginated(cfg: Config, path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    separator = "&" if "?" in path else "?"
    page = 1
    while True:
        batch = gitlab_api_json(cfg, f"{path}{separator}per_page=100&page={page}")
        if not isinstance(batch, list):
            raise SystemExit(f"Unexpected paginated GitLab API response for {path}: {batch!r}")
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items


def is_draft(mr: dict[str, Any]) -> bool:
    return bool(mr.get("draft") or mr.get("work_in_progress"))


def list_open_mrs(cfg: Config) -> list[dict[str, Any]]:
    mrs: list[dict[str, Any]] = []
    page = 1
    while True:
        path = (
            f"/projects/{cfg.project}/merge_requests"
            f"?state=opened&per_page=100&page={page}&order_by=created_at&sort=asc"
        )
        batch = gitlab_api_json(cfg, path)
        if not isinstance(batch, list):
            raise SystemExit(f"Unexpected GitLab API response for page {page}: {batch!r}")
        if not batch:
            break
        mrs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return mrs


def matching_ai_mrs(cfg: Config, *, skip_drafts: bool) -> list[dict[str, Any]]:
    matched = []
    for mr in list_open_mrs(cfg):
        source_branch = str(mr.get("source_branch") or "")
        if not source_branch.startswith(cfg.source_prefixes):
            continue
        if skip_drafts and is_draft(mr):
            continue
        matched.append(mr)
    return matched


def print_mr_table(mrs: list[dict[str, Any]]) -> None:
    if not mrs:
        print("No matching merge requests.")
        return
    print("IID   target        source                                      pipeline  title")
    print("----  ------------  ------------------------------------------  --------  -----")
    for mr in mrs:
        pipeline = mr.get("head_pipeline") or {}
        status = pipeline.get("status") or "-"
        title = str(mr.get("title") or "").replace("\n", " ")
        print(
            f"!{mr['iid']:<3}  "
            f"{mr.get('target_branch', '-'):<12}  "
            f"{mr.get('source_branch', '-'):<42}  "
            f"{status:<8}  "
            f"{title[:90]}"
        )


def cmd_ensure(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    ensure_branch(cfg)
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    sync_branch(cfg, push=args.push)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    mrs = matching_ai_mrs(cfg, skip_drafts=args.skip_drafts)
    if args.json:
        print(json.dumps(mrs, indent=2, sort_keys=True))
    else:
        print_mr_table(mrs)
    return 0


def cmd_retarget(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    mrs = matching_ai_mrs(cfg, skip_drafts=args.skip_drafts)
    changed = 0
    for mr in mrs:
        iid = mr["iid"]
        source = mr.get("source_branch")
        current_target = mr.get("target_branch")
        if not args.all_targets and current_target != args.from_target:
            continue
        if current_target == cfg.branch:
            continue

        print(f"!{iid}: {source} {current_target} -> {cfg.branch}")
        changed += 1
        if cfg.dry_run:
            continue
        gitlab_api_json(
            cfg,
            f"/projects/{cfg.project}/merge_requests/{iid}",
            method="PUT",
            fields={"target_branch": cfg.branch, "add_labels": AI_STAGING_MR_LABELS},
        )

    print(f"retargeted {changed} merge request(s)" if not cfg.dry_run else f"would retarget {changed} merge request(s)")
    return 0


def cmd_full_cycle(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    sync_branch(cfg, push=args.push)
    if args.retarget:
        return cmd_retarget(args)
    print("retarget step skipped; pass --retarget to update matching merge requests")
    return 0


def branch_has_tree_diff(cfg: Config, target_branch: str, *, source_branch: str | None = None) -> bool:
    source_branch = source_branch or cfg.branch
    fetch_branch(cfg, target_branch)
    fetch_branch(cfg, source_branch)
    result = run(
        ["git", "diff", "--quiet", remote_ref(cfg, target_branch), remote_ref(cfg, source_branch)],
        check=False,
    )
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    raise SystemExit(f"git diff failed with exit code {result.returncode}")


def branch_contains_target(cfg: Config, target_branch: str, *, source_branch: str | None = None) -> bool:
    source_branch = source_branch or cfg.branch
    result = run(
        ["git", "merge-base", "--is-ancestor", remote_ref(cfg, target_branch), remote_ref(cfg, source_branch)],
        check=False,
    )
    return result.returncode == 0


def git_lines(args: list[str]) -> list[str]:
    output = git(args)
    return output.splitlines() if output else []


def limited_lines(lines: list[str], *, limit: int) -> list[str]:
    if len(lines) <= limit:
        return lines
    omitted = len(lines) - limit
    return [*lines[:limit], f"... ({omitted} more line(s) omitted)"]


def status_counts(name_status_lines: list[str]) -> dict[str, int]:
    counts = {"added": 0, "modified": 0, "deleted": 0, "renamed": 0, "other": 0}
    for line in name_status_lines:
        status = line.split("\t", 1)[0]
        if status == "A":
            counts["added"] += 1
        elif status == "M":
            counts["modified"] += 1
        elif status == "D":
            counts["deleted"] += 1
        elif status.startswith("R"):
            counts["renamed"] += 1
        else:
            counts["other"] += 1
    return counts


def promotion_change_summary(cfg: Config, target_branch: str, *, source_branch: str | None = None) -> dict[str, Any]:
    source_branch = source_branch or cfg.branch
    base = remote_ref(cfg, target_branch)
    head = remote_ref(cfg, source_branch)
    rev_range = f"{base}..{head}"
    symmetric_range = f"{base}...{head}"

    target_sha = git(["rev-parse", "--short=12", base])
    source_sha = git(["rev-parse", "--short=12", head])
    commits = git_lines(
        [
            "log",
            "--cherry-pick",
            "--right-only",
            "--no-merges",
            "--pretty=format:%h %s",
            symmetric_range,
        ]
    )
    name_status = git_lines(["diff", "--name-status", "--find-renames", rev_range])
    diffstat = git_lines(["diff", "--stat", "--find-renames", rev_range])
    counts = status_counts(name_status)

    return {
        "target_sha": target_sha,
        "source_sha": source_sha,
        "commits": commits,
        "name_status": name_status,
        "diffstat": diffstat,
        "counts": counts,
    }


def find_promotion_mr(cfg: Config, target_branch: str, *, source_branch: str | None = None) -> dict[str, Any] | None:
    source_branch = source_branch or cfg.branch
    query = urllib.parse.urlencode(
        {
            "state": "opened",
            "source_branch": source_branch,
            "target_branch": target_branch,
            "per_page": "1",
        }
    )
    response = gitlab_api_json(cfg, f"/projects/{cfg.project}/merge_requests?{query}")
    if not isinstance(response, list):
        raise SystemExit(f"Unexpected GitLab API response while finding promotion MR: {response!r}")
    return response[0] if response else None


def promotion_description(
    cfg: Config,
    target_branch: str,
    *,
    is_up_to_date: bool,
    source_branch: str | None = None,
    summary_source_branch: str | None = None,
    rotation: dict[str, str] | None = None,
) -> str:
    source_branch = source_branch or cfg.branch
    summary = promotion_change_summary(cfg, target_branch, source_branch=summary_source_branch or source_branch)
    counts = summary["counts"]
    commits = limited_lines(summary["commits"], limit=40)
    changed_paths = limited_lines(summary["name_status"], limit=80)
    diffstat = limited_lines(summary["diffstat"], limit=80)
    status = (
        f"`{remote_ref(cfg, source_branch)}` contains `{remote_ref(cfg, target_branch)}`."
        if is_up_to_date
        else f"WARNING: `{remote_ref(cfg, source_branch)}` does not contain `{remote_ref(cfg, target_branch)}`. "
        "Review carefully before merging."
    )
    commit_block = "\n".join(f"- `{line.split(' ', 1)[0]}` {line.split(' ', 1)[1] if ' ' in line else ''}" for line in commits)
    if not commit_block:
        commit_block = "- No unique non-merge commit subjects found; review the tree diff below."
    changed_paths_block = "\n".join(changed_paths) if changed_paths else "No changed paths."
    diffstat_block = "\n".join(diffstat) if diffstat else "No diffstat."

    rotation_lines = []
    if rotation:
        rotation_lines = [
            "",
            "Rotation state:",
            "",
            f"- Snapshot branch: `{remote_ref(cfg, source_branch)}`",
            f"- Snapshot was cut from `{remote_ref(cfg, cfg.branch)}` @ `{rotation['staging_sha']}`",
            f"- `{remote_ref(cfg, cfg.branch)}` was reset to `{remote_ref(cfg, target_branch)}` @ `{rotation['target_sha']}` for future AI MRs",
        ]

    return "\n".join(
        [
            f"Promotion MR from `{source_branch}` to `{target_branch}` for human review.",
            "",
            f"This MR does not auto-merge. It promotes the current AI staging tree into `{target_branch}` after the full MR pipeline is green.",
            "",
            "Branch state:",
            "",
            f"- Source: `{remote_ref(cfg, source_branch)}` @ `{summary['source_sha']}`",
            f"- Target: `{remote_ref(cfg, target_branch)}` @ `{summary['target_sha']}`",
            f"- Up to date with target: {'yes' if is_up_to_date else 'no'}",
            *rotation_lines,
            "",
            status,
            "",
            "What is being promoted:",
            "",
            commit_block,
            "",
            "Net file change summary:",
            "",
            f"- Added: {counts['added']}",
            f"- Modified: {counts['modified']}",
            f"- Deleted: {counts['deleted']}",
            f"- Renamed: {counts['renamed']}",
            f"- Other: {counts['other']}",
            "",
            "Changed paths:",
            "",
            "```text",
            changed_paths_block,
            "```",
            "",
            "Diffstat:",
            "",
            "```text",
            diffstat_block,
            "```",
            "",
            "Review checklist:",
            "",
            "- Full MR pipeline is green.",
            "- Diff contains only expected AI staging changes.",
            f"- `{source_branch}` is up to date with `{target_branch}` before merge.",
            "- Individual AI-generated MRs included in this promotion have task scope, verification, and risk notes in their descriptions.",
            "",
            "Reviewer note: the tree diff and changed paths above are authoritative. Commit subjects are included as a readable summary, but may include staging-history commits when earlier promotions were squashed.",
        ]
    )


def timestamped_snapshot_branch(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix.rstrip('-')}-{timestamp}"


def push_snapshot_branch(cfg: Config, snapshot_branch: str, *, source_branch: str) -> None:
    source_ref = f"refs/remotes/{cfg.remote}/{source_branch}:refs/heads/{snapshot_branch}"
    run(["git", "push", cfg.remote, source_ref], dry_run=cfg.dry_run)
    action = "would create" if cfg.dry_run else "created"
    print(f"{action} snapshot branch {cfg.remote}/{snapshot_branch} from {cfg.remote}/{source_branch}")


def reset_remote_branch_to_target(
    cfg: Config,
    *,
    branch: str,
    target_branch: str,
    expected_old_sha: str,
) -> None:
    target_ref = f"refs/remotes/{cfg.remote}/{target_branch}:refs/heads/{branch}"
    lease = f"--force-with-lease=refs/heads/{branch}:{expected_old_sha}"
    run(["git", "push", lease, cfg.remote, target_ref], dry_run=cfg.dry_run)
    action = "would reset" if cfg.dry_run else "reset"
    print(f"{action} {cfg.remote}/{branch} to {cfg.remote}/{target_branch}")


def create_or_update_promotion_mr(
    cfg: Config,
    *,
    source_branch: str,
    target_branch: str,
    title: str,
    description: str,
    labels: str | None = AI_PROMOTION_MR_LABELS,
) -> dict[str, Any] | None:
    if cfg.dry_run:
        print(f"would create or update promotion MR from {source_branch} to {target_branch}: {title}")
        return None

    existing = find_promotion_mr(cfg, target_branch, source_branch=source_branch)
    if existing:
        fields = {
            "title": title,
            "description": description,
            "remove_source_branch": "false",
            "squash": "false",
        }
        if labels:
            fields["add_labels"] = labels
        updated = gitlab_api_json(
            cfg,
            f"/projects/{cfg.project}/merge_requests/{existing['iid']}",
            method="PUT",
            fields=fields,
        )
        print(f"Updated promotion MR: !{existing['iid']} {existing['web_url']}")
        return updated if isinstance(updated, dict) else existing

    fields = {
        "source_branch": source_branch,
        "target_branch": target_branch,
        "title": title,
        "description": description,
        "remove_source_branch": "false",
        "squash": "false",
    }
    if labels:
        fields["labels"] = labels
    mr = gitlab_api_json(cfg, f"/projects/{cfg.project}/merge_requests", method="POST", fields=fields)
    if not isinstance(mr, dict):
        raise SystemExit(f"Unexpected GitLab API response while creating promotion MR: {mr!r}")
    print(f"Created promotion MR: !{mr['iid']} {mr['web_url']}")
    return mr


def cmd_promote(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    target_branch = args.target_branch
    if not branch_has_tree_diff(cfg, target_branch):
        print(f"No tree diff between {cfg.remote}/{target_branch} and {cfg.remote}/{cfg.branch}; no MR needed.")
        return 0

    is_up_to_date = branch_contains_target(cfg, target_branch)
    description = promotion_description(cfg, target_branch, is_up_to_date=is_up_to_date)
    create_or_update_promotion_mr(
        cfg,
        source_branch=cfg.branch,
        target_branch=target_branch,
        title=f"chore: promote {cfg.branch} to {target_branch}",
        description=description,
    )
    return 0


def cmd_rotate_promotion(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    target_branch = args.target_branch
    assert_clean_worktree()
    ensure_branch(cfg, source_branch=target_branch)
    has_tree_diff = branch_has_tree_diff(cfg, target_branch)

    staging_ref = remote_ref(cfg, cfg.branch)
    target_ref = remote_ref(cfg, target_branch)
    staging_sha = git(["rev-parse", staging_ref])
    target_sha = git(["rev-parse", target_ref])

    if not has_tree_diff:
        if staging_sha != target_sha:
            reset_remote_branch_to_target(
                cfg,
                branch=cfg.branch,
                target_branch=target_branch,
                expected_old_sha=staging_sha,
            )
        else:
            print(f"{staging_ref} already matches {target_ref}; no promotion MR needed.")
        return 0

    snapshot_branch = args.snapshot_branch or timestamped_snapshot_branch(args.snapshot_prefix)
    if remote_branch_exists(cfg, snapshot_branch):
        raise SystemExit(f"Snapshot branch already exists: {cfg.remote}/{snapshot_branch}")

    push_snapshot_branch(cfg, snapshot_branch, source_branch=cfg.branch)
    summary_source_branch = cfg.branch if cfg.dry_run else snapshot_branch
    if not cfg.dry_run:
        fetch_branch(cfg, snapshot_branch)

    reset_remote_branch_to_target(
        cfg,
        branch=cfg.branch,
        target_branch=target_branch,
        expected_old_sha=staging_sha,
    )

    is_up_to_date = branch_contains_target(cfg, target_branch, source_branch=summary_source_branch)
    description = promotion_description(
        cfg,
        target_branch,
        source_branch=snapshot_branch,
        summary_source_branch=summary_source_branch,
        is_up_to_date=is_up_to_date,
        rotation={"staging_sha": staging_sha[:12], "target_sha": target_sha[:12]},
    )
    create_or_update_promotion_mr(
        cfg,
        source_branch=snapshot_branch,
        target_branch=target_branch,
        title=f"chore: promote {snapshot_branch} to {target_branch}",
        description=description,
    )
    return 0


def normalized_snapshot_prefix(prefix: str) -> str:
    return f"{prefix.rstrip('-')}-"


def list_open_promotion_mrs(cfg: Config, *, target_branch: str, snapshot_prefix: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "state": "opened",
            "target_branch": target_branch,
            "order_by": "created_at",
            "sort": "asc",
        }
    )
    mrs = gitlab_paginated(cfg, f"/projects/{cfg.project}/merge_requests?{query}")
    return [
        mr
        for mr in mrs
        if str(mr.get("source_branch") or "").startswith(normalized_snapshot_prefix(snapshot_prefix))
    ]


def failed_jobs_for_pipeline(cfg: Config, pipeline_id: int) -> list[dict[str, Any]]:
    jobs = gitlab_paginated(cfg, f"/projects/{cfg.project}/pipelines/{pipeline_id}/jobs")
    return [job for job in jobs if job.get("status") in FAILED_PIPELINE_STATUSES]


def format_failed_jobs(jobs: list[dict[str, Any]], *, limit: int = 10) -> str:
    if not jobs:
        return "no failed jobs found"
    lines = []
    for job in jobs[:limit]:
        lines.append(
            f"{job.get('stage', '-')}/{job.get('name', '-')} "
            f"status={job.get('status', '-')} "
            f"url={job.get('web_url', '-')}"
        )
    if len(jobs) > limit:
        lines.append(f"... {len(jobs) - limit} more failed job(s)")
    return "\n".join(lines)


def safe_tmp_prefix(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)


def clean_rebase_promotion_branch(cfg: Config, *, source_branch: str, target_branch: str) -> bool:
    fetch_branch(cfg, target_branch)
    fetch_branch(cfg, source_branch)
    old_sha = git(["rev-parse", remote_ref(cfg, source_branch)])
    worktree_path = tempfile.mkdtemp(prefix=f"{safe_tmp_prefix(source_branch)}-rebase-")
    keep_worktree = False
    added = run(
        ["git", "worktree", "add", "--detach", worktree_path, remote_ref(cfg, source_branch)],
        check=False,
    )
    if added.returncode != 0:
        shutil.rmtree(worktree_path, ignore_errors=True)
        details = (added.stderr or added.stdout or "").strip()
        raise SystemExit(f"Could not create temporary promotion rebase worktree at {worktree_path}.\n{details}")

    try:
        rebased = run(["git", "rebase", remote_ref(cfg, target_branch)], check=False, cwd=worktree_path)
        if rebased.returncode != 0:
            run(["git", "rebase", "--abort"], check=False, cwd=worktree_path)
            print(f"! promotion rebase conflict: {cfg.remote}/{source_branch} onto {cfg.remote}/{target_branch}")
            return False

        lease = f"--force-with-lease=refs/heads/{source_branch}:{old_sha}"
        run(
            ["git", "push", lease, cfg.remote, f"HEAD:refs/heads/{source_branch}"],
            dry_run=cfg.dry_run,
            cwd=worktree_path,
        )
        action = "would rebase" if cfg.dry_run else "rebased"
        print(f"{action} {cfg.remote}/{source_branch} onto {cfg.remote}/{target_branch}")
        return True
    except BaseException:
        keep_worktree = True
        print(f"Preserving failed promotion rebase worktree for inspection: {worktree_path}", file=sys.stderr)
        raise
    finally:
        if not keep_worktree:
            run(["git", "worktree", "remove", "--force", worktree_path], check=False)
            shutil.rmtree(worktree_path, ignore_errors=True)


def cmd_babysit_promotion(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    target_branch = args.target_branch
    assert_clean_worktree()
    fetch_branch(cfg, target_branch)
    promotion_mrs = list_open_promotion_mrs(
        cfg,
        target_branch=target_branch,
        snapshot_prefix=args.snapshot_prefix,
    )
    if not promotion_mrs:
        print(f"No open promotion MRs targeting {target_branch} with prefix {normalized_snapshot_prefix(args.snapshot_prefix)}")
        return 0

    actions = 0
    for mr in promotion_mrs:
        source_branch = str(mr.get("source_branch") or "")
        iid = int(mr["iid"])
        pipeline = mr.get("head_pipeline") or {}
        pipeline_status = str(pipeline.get("status") or "none")
        pipeline_id = pipeline.get("id")

        if not remote_branch_exists(cfg, source_branch):
            print(f"!{iid}: source branch missing: {cfg.remote}/{source_branch}")
            continue

        fetch_branch(cfg, source_branch)
        up_to_date = branch_contains_target(cfg, target_branch, source_branch=source_branch)
        print(
            f"!{iid}: source={source_branch} pipeline={pipeline_status} "
            f"up_to_date_with_{target_branch}={'yes' if up_to_date else 'no'} {mr.get('web_url', '')}"
        )

        if not up_to_date and pipeline_status not in ACTIVE_PIPELINE_STATUSES:
            if actions >= args.max_rebases:
                print(f"!{iid}: skipped rebase; max rebases reached for this cycle")
                continue
            rebased = clean_rebase_promotion_branch(
                cfg,
                source_branch=source_branch,
                target_branch=target_branch,
            )
            actions += 1
            if rebased:
                fetch_branch(cfg, source_branch)
                description = promotion_description(
                    cfg,
                    target_branch,
                    source_branch=source_branch,
                    is_up_to_date=True,
                )
                create_or_update_promotion_mr(
                    cfg,
                    source_branch=source_branch,
                    target_branch=target_branch,
                    title=f"chore: promote {source_branch} to {target_branch}",
                    description=description,
                )
            else:
                print(f"!{iid}: needs human rebase/conflict review")
            continue

        if pipeline_status in ACTIVE_PIPELINE_STATUSES:
            print(f"!{iid}: waiting for full MR pipeline")
        elif pipeline_status == "success":
            print(f"!{iid}: ready for human review")
        elif pipeline_status in FAILED_PIPELINE_STATUSES:
            if pipeline_id is None:
                print(f"!{iid}: promotion pipeline {pipeline_status}; no pipeline id available")
            else:
                jobs = failed_jobs_for_pipeline(cfg, int(pipeline_id))
                print(
                    f"!{iid}: promotion pipeline {pipeline_status}; "
                    f"repair {source_branch} and push a new full pipeline. Failed jobs:\n{format_failed_jobs(jobs)}"
                )
        else:
            print(f"!{iid}: promotion pipeline status is {pipeline_status}; no action")

    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote", default=os.environ.get("AI_STAGING_REMOTE", DEFAULT_REMOTE))
    parser.add_argument("--branch", default=os.environ.get("AI_STAGING_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--project", help="GitLab project path or numeric project id")
    parser.add_argument(
        "--source-prefix",
        action="append",
        help="AI source branch prefix to include. May be repeated. Default: ai-task-",
    )
    parser.add_argument("--dry-run", action="store_true", help="print writes without performing them")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure = subparsers.add_parser("ensure-branch", help="create the origin AI staging branch if missing")
    ensure.set_defaults(func=cmd_ensure)

    sync = subparsers.add_parser("sync-branch", help="merge origin/master into the AI staging branch")
    sync.add_argument("--push", action="store_true", help="push the synced branch to origin")
    sync.set_defaults(func=cmd_sync)

    list_cmd = subparsers.add_parser("list", help="list open AI-generated merge requests")
    list_cmd.add_argument("--json", action="store_true", help="emit raw JSON")
    list_cmd.add_argument("--skip-drafts", action="store_true", help="exclude draft merge requests")
    list_cmd.set_defaults(func=cmd_list)

    retarget = subparsers.add_parser("retarget", help="retarget matching AI merge requests to the staging branch")
    retarget.add_argument("--from-target", default="master", help="only retarget MRs currently targeting this branch")
    retarget.add_argument("--all-targets", action="store_true", help="retarget regardless of current target branch")
    retarget.add_argument("--skip-drafts", action="store_true", help="exclude draft merge requests")
    retarget.set_defaults(func=cmd_retarget)

    full_cycle = subparsers.add_parser("full-cycle", help="sync branch and optionally retarget matching MRs")
    full_cycle.add_argument("--push", action="store_true", help="push the synced branch to origin")
    full_cycle.add_argument("--retarget", action="store_true", help="retarget matching MRs after syncing")
    full_cycle.add_argument("--from-target", default="master", help="only retarget MRs currently targeting this branch")
    full_cycle.add_argument("--all-targets", action="store_true", help="retarget regardless of current target branch")
    full_cycle.add_argument("--skip-drafts", action="store_true", help="exclude draft merge requests")
    full_cycle.set_defaults(func=cmd_full_cycle)

    promote = subparsers.add_parser("promote", help="file an ai-staging to master promotion MR for human review")
    promote.add_argument(
        "--require-api-token",
        action="store_true",
        help="require AI_STAGING_BOT_TOKEN and never fall back to local glab authentication",
    )
    promote.add_argument("--target-branch", default=os.environ.get("AI_STAGING_PROMOTION_TARGET", "master"))
    promote.set_defaults(func=cmd_promote)

    rotate = subparsers.add_parser(
        "rotate-promotion",
        help="snapshot ai-staging to a timestamped branch, reset ai-staging to master, and file a promotion MR",
    )
    rotate.add_argument(
        "--require-api-token",
        action="store_true",
        help="require AI_STAGING_BOT_TOKEN and never fall back to local glab authentication",
    )
    rotate.add_argument("--target-branch", default=os.environ.get("AI_STAGING_PROMOTION_TARGET", "master"))
    rotate.add_argument("--snapshot-prefix", default=DEFAULT_PROMOTION_PREFIX)
    rotate.add_argument("--snapshot-branch", help="explicit snapshot branch name; default is <prefix>-YYYYmmdd-HHMMSS UTC")
    rotate.set_defaults(func=cmd_rotate_promotion)

    babysit = subparsers.add_parser(
        "babysit-promotion",
        help="watch open promotion MRs, clean-rebase outdated snapshot branches, and report full-CI status",
    )
    babysit.add_argument(
        "--require-api-token",
        action="store_true",
        help="require AI_STAGING_BOT_TOKEN and never fall back to local glab authentication",
    )
    babysit.add_argument("--target-branch", default=os.environ.get("AI_STAGING_PROMOTION_TARGET", "master"))
    babysit.add_argument("--snapshot-prefix", default=DEFAULT_PROMOTION_PREFIX)
    babysit.add_argument(
        "--max-rebases",
        type=int,
        default=1,
        help="maximum clean promotion-branch rebases to push in one cycle",
    )
    babysit.set_defaults(func=cmd_babysit_promotion)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

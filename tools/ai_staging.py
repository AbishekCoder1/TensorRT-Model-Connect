#!/usr/bin/env python3
"""Maintain the AI staging branch and route AI-generated merge requests."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Any


DEFAULT_BRANCH = "ai-staging"
DEFAULT_REMOTE = "origin"
DEFAULT_SOURCE_PREFIXES = ("agent-2-",)


@dataclass(frozen=True)
class Config:
    remote: str
    branch: str
    project: str
    source_prefixes: tuple[str, ...]
    dry_run: bool


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+ " + shlex.join(cmd), file=sys.stderr)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def git(args: list[str], *, check: bool = True, dry_run: bool = False) -> str:
    result = run(["git", *args], check=check, capture=True, dry_run=dry_run)
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
    )


def fetch_branch(cfg: Config, branch: str) -> None:
    git(
        [
            "fetch",
            cfg.remote,
            f"+refs/heads/{branch}:refs/remotes/{cfg.remote}/{branch}",
        ]
    )


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
            "Refusing to sync with local changes present. Commit, stash, or use a clean worktree.\n"
            + status
        )


def ensure_branch(cfg: Config) -> bool:
    fetch_branch(cfg, "master")
    if remote_branch_exists(cfg, cfg.branch):
        print(f"{cfg.remote} branch exists: {cfg.branch}")
        fetch_branch(cfg, cfg.branch)
        return False

    source = f"refs/remotes/{cfg.remote}/master:refs/heads/{cfg.branch}"
    run(["git", "push", cfg.remote, source], dry_run=cfg.dry_run)
    print(f"created {cfg.remote}/{cfg.branch} from {cfg.remote}/master")
    return True


def sync_branch(cfg: Config, *, push: bool) -> None:
    assert_clean_worktree()
    ensure_branch(cfg)

    if local_branch_exists(cfg.branch):
        git(["switch", cfg.branch], dry_run=cfg.dry_run)
        git(["merge", "--ff-only", f"{cfg.remote}/{cfg.branch}"], dry_run=cfg.dry_run)
    else:
        git(["switch", "--track", "-c", cfg.branch, f"{cfg.remote}/{cfg.branch}"], dry_run=cfg.dry_run)

    contains_master = run(
        ["git", "merge-base", "--is-ancestor", f"{cfg.remote}/master", "HEAD"],
        check=False,
        dry_run=cfg.dry_run,
    )
    if contains_master.returncode == 0:
        print(f"{cfg.branch} already contains {cfg.remote}/master")
    else:
        git(["merge", "--no-edit", f"{cfg.remote}/master"], dry_run=cfg.dry_run)

    if push:
        run(["git", "push", cfg.remote, f"HEAD:refs/heads/{cfg.branch}"], dry_run=cfg.dry_run)


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
        batch = glab_api_json(path)
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
        glab_api_json(
            f"/projects/{cfg.project}/merge_requests/{iid}",
            method="PUT",
            fields={"target_branch": cfg.branch},
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


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote", default=os.environ.get("AI_STAGING_REMOTE", DEFAULT_REMOTE))
    parser.add_argument("--branch", default=os.environ.get("AI_STAGING_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--project", help="GitLab project path or numeric project id")
    parser.add_argument(
        "--source-prefix",
        action="append",
        help="AI source branch prefix to include. May be repeated. Default: agent-2-",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

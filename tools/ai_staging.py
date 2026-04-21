#!/usr/bin/env python3
"""Maintain the AI staging branch and route AI-generated merge requests."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
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
    require_api_token: bool


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


def branch_has_tree_diff(cfg: Config, target_branch: str) -> bool:
    fetch_branch(cfg, target_branch)
    fetch_branch(cfg, cfg.branch)
    result = run(
        ["git", "diff", "--quiet", f"{cfg.remote}/{target_branch}", f"{cfg.remote}/{cfg.branch}"],
        check=False,
    )
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    raise SystemExit(f"git diff failed with exit code {result.returncode}")


def branch_contains_target(cfg: Config, target_branch: str) -> bool:
    result = run(
        ["git", "merge-base", "--is-ancestor", f"{cfg.remote}/{target_branch}", f"{cfg.remote}/{cfg.branch}"],
        check=False,
    )
    return result.returncode == 0


def find_promotion_mr(cfg: Config, target_branch: str) -> dict[str, Any] | None:
    query = urllib.parse.urlencode(
        {
            "state": "opened",
            "source_branch": cfg.branch,
            "target_branch": target_branch,
            "per_page": "1",
        }
    )
    response = gitlab_api_json(cfg, f"/projects/{cfg.project}/merge_requests?{query}")
    if not isinstance(response, list):
        raise SystemExit(f"Unexpected GitLab API response while finding promotion MR: {response!r}")
    return response[0] if response else None


def promotion_description(cfg: Config, target_branch: str, *, is_up_to_date: bool) -> str:
    status = (
        f"`{cfg.remote}/{cfg.branch}` contains `{cfg.remote}/{target_branch}`."
        if is_up_to_date
        else f"WARNING: `{cfg.remote}/{cfg.branch}` does not contain `{cfg.remote}/{target_branch}`. "
        "Review carefully before merging."
    )
    return "\n".join(
        [
            f"Scheduled promotion MR from `{cfg.branch}` to `{target_branch}`.",
            "",
            "This MR is filed automatically for human review. It does not auto-merge.",
            "",
            status,
            "",
            "Review checklist:",
            "",
            "- Full MR pipeline is green.",
            "- Diff contains only expected AI staging changes.",
            f"- `{cfg.branch}` is up to date with `{target_branch}` before merge.",
        ]
    )


def cmd_promote(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    target_branch = args.target_branch
    if not branch_has_tree_diff(cfg, target_branch):
        print(f"No tree diff between {cfg.remote}/{target_branch} and {cfg.remote}/{cfg.branch}; no MR needed.")
        return 0

    is_up_to_date = branch_contains_target(cfg, target_branch)
    existing = find_promotion_mr(cfg, target_branch)
    if existing:
        print(f"Promotion MR already exists: !{existing['iid']} {existing['web_url']}")
        return 0

    fields = {
        "source_branch": cfg.branch,
        "target_branch": target_branch,
        "title": f"chore: promote {cfg.branch} to {target_branch}",
        "description": promotion_description(cfg, target_branch, is_up_to_date=is_up_to_date),
        "remove_source_branch": "false",
        "squash": "false",
    }
    mr = gitlab_api_json(cfg, f"/projects/{cfg.project}/merge_requests", method="POST", fields=fields)
    if cfg.dry_run:
        print(f"would create promotion MR from {cfg.branch} to {target_branch}")
        return 0
    if not isinstance(mr, dict):
        raise SystemExit(f"Unexpected GitLab API response while creating promotion MR: {mr!r}")
    print(f"Created promotion MR: !{mr['iid']} {mr['web_url']}")
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

    promote = subparsers.add_parser("promote", help="file an ai-staging to master promotion MR for human review")
    promote.add_argument(
        "--require-api-token",
        action="store_true",
        help="require AI_STAGING_BOT_TOKEN and never fall back to local glab authentication",
    )
    promote.add_argument("--target-branch", default=os.environ.get("AI_STAGING_PROMOTION_TARGET", "master"))
    promote.set_defaults(func=cmd_promote)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

---
name: submit-gitlab-mr
description: Use when pushing a branch and creating a GitLab merge request. Trigger when user says "submit MR", "create MR", "open merge request", or "push and create MR". Requires glab CLI.
---

# Submit GitLab MR

## Overview

Use `glab` (not `gh`) to create GitLab merge requests. The `glab mr create` command requires `dangerouslyDisableSandbox: true` because it writes to `.git/config`.

## Quick Reference

```bash
# Push branch first
git push -u origin <branch-name>

# Create MR (requires dangerouslyDisableSandbox: true)
glab mr create \
  --source-branch <branch> \
  --target-branch master \
  --title "feat: short title" \
  --description "$(cat <<'EOF'
## Summary
- bullet points

## Test plan
- [x] done item
- [ ] todo item
EOF
)" \
  --remove-source-branch
```

## Key Flags

| Flag | Purpose |
|------|---------|
| `-s, --source-branch` | Branch to merge from |
| `-b, --target-branch` | Branch to merge into (default: project default) |
| `-t, --title` | MR title |
| `-d, --description` | MR body (use heredoc for multiline) |
| `--remove-source-branch` | Delete branch after merge |
| `--draft` | Mark as draft/WIP |
| `-a, --assignee` | Assign by username |
| `--reviewer` | Request review by username |
| `-l, --label` | Add labels (comma-separated) |
| `-f, --fill` | Auto-fill title/description from commits |

## Common Mistakes

- **Using `gh` instead of `glab`**: This is a GitLab project. `gh` is for GitHub.
- **Running without `dangerouslyDisableSandbox: true`**: `glab` writes to `.git/config`, which the sandbox blocks. You'll get "Device or resource busy" errors.
- **Forgetting to push first**: `glab mr create` needs the branch on the remote. Push with `-u` before creating.
- **Heredoc quoting**: Use `'EOF'` (quoted) to prevent shell expansion in the description body.

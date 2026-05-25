# GitHub Public Publish v1

## Scope

- task_id=github_public_publish_v1
- repo_name=ChessMachineZero
- repo_url=https://github.com/TryDotAtwo/ChessMachineZero
- visibility=public
- branch=main
- initial_commit=cbdee31

## Preflight

```text
gh --version
gh version 2.89.0 (2026-03-26)
```

```text
gh auth status
github.com
  ✓ Logged in to github.com account TryDotAtwo (keyring)
  - Active account: true
  - Git operations protocol: https
  - Token scopes: 'gist', 'read:org', 'repo', 'workflow'
```

```text
publish_scope=129 tracked files, 643662 bytes excluding ignored artifacts
ignored=.playwright-mcp/, .pytest_cache/, __pycache__/, *.pyc, *.png
secret_scan=0 matches for public-token/API-key/secret/password patterns outside excluded historical docs/test-results
```

## Git Commands

```text
git init -b main
Initialized empty Git repository in C:/Users/Иван Литвак/Documents/ChessMachineZero/.git/
```

```text
git commit -m "Initial ChessMachineZero Percepta trace VM"
[main (root-commit) cbdee31] Initial ChessMachineZero Percepta trace VM
129 files changed, 14222 insertions(+)
```

```text
gh repo create ChessMachineZero --public --description "Percepta-style trace VM chess machine with frozen attention rule weights" --source . --remote origin --push
https://github.com/TryDotAtwo/ChessMachineZero
branch 'main' set up to track 'origin/main'.
To https://github.com/TryDotAtwo/ChessMachineZero.git
 * [new branch]      HEAD -> main
```

## Result

- status=published
- repo_url=https://github.com/TryDotAtwo/ChessMachineZero
- visibility=public
- local_branch=main
- remote=origin
- remote_tracking=origin/main

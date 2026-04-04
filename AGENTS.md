# Repository Agent Instructions

These instructions apply to Codex and similar code agents working in this
repository.

## Git Workflow

- Stay on `main` by default.
- Do not create or switch to feature branches unless the user explicitly asks.
- Do not open a pull request unless the user explicitly asks.
- If the user asks to commit and push, commit directly to `main` and push
  `origin main`.
- Before committing, inspect `git status --short` and stage only the files that
  belong to the requested task.
- Do not rewrite history, force-push, or amend published commits unless the user
  explicitly asks.

## Change Scope

- Keep commits narrowly scoped to the task at hand.
- Do not include unrelated local changes in a commit.
- If the worktree is mixed and the intended scope is unclear, stop and ask.

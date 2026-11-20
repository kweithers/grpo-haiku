# Workspace Guidelines

## Git Workflow: Small, Frequent Commits

- Make small, frequent, atomic commits as we iterate rather than batching unrelated changes together.
- Commit after each discrete, verified unit of work—for example:
  - a reward-shaping or verifier change,
  - a hyperparameter/config adjustment,
  - a bug fix or new feature,
  - before launching a new training run so the exact code behind each run is recorded in git history.
- Run the unit test suite (`.venv/bin/pytest`) before committing code changes so every commit leaves the repository in a clean, passing state.
- Use concise conventional commit messages matching this repo's history (`feat:`, `fix:`, `tune:`, `test:`, `docs:`).
- Never commit untracked training outputs, checkpoints, or virtual environments (`runs/`, `.venv/`).

## macOS Sleep Prevention (`caffeinate`)

- Always run training runs (`grpo_haiku.grpo`, `grpo_haiku.sft`) and inference/evaluation runs (`grpo_haiku.eval`, `grpo_haiku.generate`) with `caffeinate -i` (e.g., `caffeinate -i .venv/bin/python -m grpo_haiku.grpo ...`) so the Mac does not sleep mid-run.

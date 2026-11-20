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

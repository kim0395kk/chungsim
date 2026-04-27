# AGENTS.md (Codex project rules)

## Quick commands
- Install deps: `python -m pip install -r requirements.txt`
- Run app: `streamlit run streamlit_app.py`
- Compile check: `python -m compileall .`
- Run tests: `python -m pytest -q`
- Lint (if installed): `python -m ruff check .`
- Format check (if installed): `python -m black --check .`

## Rules
- Prefer small, safe commits.
- Update tests/docs when changing behavior.
- Do not commit secrets. Use .env.example if needed.

## Sandbox & workflow
- This repository runs in a sandbox: file edits are fully allowed inside this repo.
- Preferred execution flow for upgrades/fixes:
1. Plan
2. Implement in small, safe steps
3. Verify (`compileall`, targeted tests, app boot)
4. Review and summarize remaining risks/TODOs
- Keep app runnable after each step when practical.
- Use safe defaults when keys are missing (no hard failure on boot).

## Repo notes
- Skills live in .agents/skills/*/SKILL.md

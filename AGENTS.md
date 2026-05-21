# AGENTS.md

## Environment

- Use `uv` for dependency management.
- The project virtual environment should live at `.venv/` in the repo root.
- Prefer `uv add <package>` over `pip install <package>`.
- Run Python commands through the project environment.

## Common commands

- Install dependencies: `uv sync`
- Run tests: `uv run pytest`
- Run linting: `uv run ruff check .`
- Format code: `uv run ruff format .`

## Coding style
- Prefer small, typed, testable functions.
- Add or update tests for behavior changes.
- Avoid large rewrites unless explicitly requested.
- Explain non-obvious implementation choices briefly.
- Prefer single quotes over double quotes.
- In Python, place a space between an expected parameter and value. Example: `json.dumps(data, indent = 4)`

## Safety

- Do not modify secrets, credentials, `.env`, or deployment files unless explicitly asked.
- Before deleting files, explain why.
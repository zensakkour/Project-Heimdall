# Contributing

This repo uses **trunk-based development** with `master` as the trunk.

## Ground Rules

- Be respectful and professional in all interactions (see []()).
- Keep changes focused and small whenever possible.
- Do not commit secrets, API keys, or private datasets.

## Branches
Short-lived branches only:
- `feat/<short-topic>`
- `fix/<short-topic>`
- `chore/<short-topic>`

Examples:
- `feat/geo-scoring-ui`
- `fix/geo-eval-paths`
- `chore/update-docs`

## Workflow
1. Branch off `master`.
2. Commit small, focused changes.
3. Open a pull request using the PR template.
4. Run local tests before merge (`python -m pytest -q`).
5. Merge back quickly and delete the branch.

## Commit Messages
Use Conventional Commits:
- `feat: ...`
- `fix: ...`
- `chore: ...`
- `docs: ...`
- `refactor: ...`
- `test: ...`

Examples:
- `feat: add geo scoring tab`
- `fix: resolve geo eval paths`
- `docs: update scoring instructions`

## Merging
Rebase or merge is fine. Keep history clean and avoid long-lived branches.

## Pull Requests

- Fill out `.github/pull_request_template.md`.
- Link related issues (`Closes #...`) where applicable.
- Update docs for user-visible behavior changes.
- Add or update tests when behavior changes.

## Issues

- Use the built-in issue forms under `.github/ISSUE_TEMPLATE`.
- For vulnerabilities, do not open a public issue. Follow []().

## Branch Helper Script (Windows)
Use the helper to create a dated branch name:
```powershell
.\scripts\new-branch.ps1 -Type feat -Name "geo accuracy tuning"
.\scripts\new-branch.ps1 -Type fix -Name "eval paths"
```
This creates `feat/yyyymmdd-geo-accuracy-tuning` and switches to it.

## Notes
- `master` is the default branch.
- Large datasets live outside git (see README).
- Dependency and action updates are automated through Dependabot.

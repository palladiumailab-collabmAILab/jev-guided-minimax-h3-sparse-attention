# Codex harness upstream

The shared Codex development harness in this repository is sourced from:

- repository: `palladiumailab-collabmAILab/codex-dev-harness`
- source revision: `90bcc9bd7c67c636b287a68091a8ae4c61952391`

## Ownership

The upstream repository is the canonical source for all shared harness content. The files below are upstream-managed and must not be edited independently here. Project-specific rules belong in `AGENTS.project.md` or other explicitly project-specific files.

When a shared rule needs to change, update and validate `codex-dev-harness` first, then synchronize these files from the new pinned revision.

## Upstream-managed files

- `AGENTS.md`
- `docs/project-baseline.md`
- `docs/harness-architecture.md`
- `docs/testing-governance.md`
- `skills/repo-research/SKILL.md`
- `skills/github-operations/SKILL.md`
- `skills/self-improvement/SKILL.md`
- `skills/long-running-work/SKILL.md`
- `templates/codex-progress.md`
- `templates/project-specs/README.md`

## Applied overlays

- testing governance / Sol-Luna role split: `4e0ad32b680451610b3f601e982cc5b04956bf1b`

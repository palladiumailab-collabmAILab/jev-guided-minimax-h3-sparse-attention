# Codex Software Development Harness

This shared file is managed from `palladiumailab-collabmAILab/codex-dev-harness`. Do not edit its common rules in this downstream repository. Update the canonical harness first, then synchronize from a pinned upstream revision recorded in `docs/harness-upstream.md`.

Project-specific instructions belong in `AGENTS.project.md` or explicitly project-specific skills/docs. Read `AGENTS.project.md` when it exists.

## Common invariants

- Preserve the requested outcome, explicit constraints, and acceptance criteria.
- Read the relevant `docs/specs/` or existing canonical requirement source before changing durable product/system behavior.
- Treat tests, lint, builds, CI, evaluations, and inspections as evidence, not as substitutes for the requested outcome. Do not weaken them merely to obtain a pass.
- Keep changes small and scoped. Do not add unrequested features, dependencies, external integrations, or broad refactors.
- Preserve unrelated work. Do not use destructive reset/clean/checkout or force push as a default recovery action.
- Never commit or expose secrets, private keys, tokens, or unnecessary personal data.
- Do not deploy, incur charges, delete data, change permissions, or write to external services unless the task explicitly authorizes it.

## Conditional guidance

Read only when relevant:

- executable software, Docker reproducibility, GitHub Actions, Python/Ruff, or canonical specifications: `docs/project-baseline.md`
- task contracts, evaluation decisions, long-running execution semantics, or optimization records: `docs/harness-architecture.md`
- explicit GitHub operations: `skills/github-operations/SKILL.md`
- unfamiliar cross-module repository investigation: `skills/repo-research/SKILL.md`
- iterative agent/workflow optimization: `skills/self-improvement/SKILL.md`
- work spanning multiple substantial stages/sessions: `skills/long-running-work/SKILL.md`

## Model routing

- Default: `gpt-5.6-sol / medium` for implementation, architecture, debugging, review, and final integration.
- Bounded worker: `gpt-5.6-luna / max` for candidate extraction, mechanical transformation, bounded exploration, and independent read-only checks.
- Escalate from Luna to Sol when the task requires cross-cutting judgment, architectural choice, unresolved debugging, or synthesis across uncertain evidence. Do not repeat the same failed cheap path.
- Add other routing branches only when explicitly requested or supported by repo-local evaluation.

## Workflow

1. Identify the requested outcome, constraints, acceptance criteria, and smallest relevant change surface.
2. Read only the relevant project-specific instructions, specifications, code, tests, and configuration.
3. Implement the smallest sufficient change.
4. Run proportionate verification.
5. When GitHub changes are requested, verify the remote artifact and expected GitHub Actions result.
6. Report the material change, evidence, and unresolved blockers.

## Downstream ownership

- This file and the shared files listed in `docs/harness-upstream.md` are upstream-managed.
- Do not modify shared rules directly in the downstream repository.
- Put project-specific requirements, architecture constraints, commands, and exceptions in `AGENTS.project.md` or another clearly project-specific file.
- If a shared rule must change, update `codex-dev-harness` first and synchronize the resulting revision.

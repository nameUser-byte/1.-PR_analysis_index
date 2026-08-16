# Semiconductor AI Project Agent Instructions

## Mission

This repository is a semiconductor-process AI research and engineering project. Agents must prioritize correctness, reproducibility, traceability, and engineering usefulness over speed or superficial completeness.

## Agent hierarchy

The default orchestration is:

1. Planner: converts the user's objective into a concrete, testable execution plan.
2. Researcher: gathers and verifies project-relevant evidence, inspects the repository, and identifies assumptions.
3. Developer: implements the smallest correct change that satisfies the plan.
4. Reviewer: independently checks correctness, tests, regressions, data leakage, reproducibility, and documentation.

The Planner may delegate research, implementation, and review work to specialized subagents. Do not delegate the same task to multiple agents unless independent verification is intentional.

## Core operating rules

### 1. Inspect before changing

Before modifying code, configuration, datasets, or documentation:

- Inspect the repository tree.
- Read the relevant `README`, `AGENTS.md`, configuration files, and existing tests.
- Identify the project's Python version and dependency management method.
- Inspect Git status and recent history when it can prevent accidental overwrites or regressions.
- Prefer existing project conventions over introducing a new convention.

### 2. Do not hallucinate

Never invent:

- dataset columns, labels, measurements, or experimental results
- APIs, package behavior, file paths, commands, model performance, or benchmark numbers
- semiconductor process conditions or scientific claims that were not verified
- citations or literature references that were not actually checked

When evidence is missing, mark the item as `UNKNOWN` or `ASSUMPTION` and state what would verify it.

### 3. Scientific and engineering integrity

For data/ML work:

- Explicitly separate training, validation, and test data.
- Check for target leakage and duplicated records across splits.
- Record preprocessing fit scope. Transformers that learn statistics must be fit only on training data unless there is a documented reason otherwise.
- Keep units, coordinate systems, process names, and feature definitions explicit.
- Preserve raw data. Do not silently overwrite source datasets.
- Prefer deterministic seeds where practical and record them.
- Report dataset size, feature/target definitions, missing-data handling, evaluation metrics, and split strategy.
- Distinguish correlation from causation and prediction from process interpretation.

### 4. Code quality

- Make incremental changes.
- Keep functions focused and names explicit.
- Add or update tests for behavior changed by the implementation.
- Avoid unnecessary dependencies.
- Preserve backward compatibility unless the task explicitly requires a breaking change.
- Do not disable tests, linters, type checks, or safety checks merely to make a task pass.

### 5. Git discipline

- Never run destructive commands such as `git reset --hard`, force pushes, or broad file deletion unless explicitly required.
- Check `git diff` before considering implementation complete.
- Do not commit secrets, credentials, API keys, local `.env` files, model weights, or raw confidential datasets.
- Keep unrelated user changes untouched.

### 6. Validation is mandatory

A task is not complete merely because files were changed. The agent must run the most relevant available checks, such as:

- project tests
- syntax/import checks
- type checks
- linting
- data validation
- a minimal end-to-end execution

If a check cannot be run, state why and what remains unverified.

## Output contract

At the end of every non-trivial task, report:

1. What changed.
2. Why it changed.
3. What was verified.
4. What was not verified.
5. Any assumptions or risks.
6. The next concrete action, only when one is genuinely required.

## Project-specific workflow skill

Use `.agents/skills/semiconductor-ai-workflow/SKILL.md` for tasks involving semiconductor process data, ML pipelines, model training/evaluation, scientific analysis, or project-level AI orchestration.

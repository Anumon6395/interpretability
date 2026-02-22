# PROJECT_STANDARDS.md

## Purpose

This document defines structural, naming, reproducibility, and coding standards across all projects in this repository.

Goals:

- Maintain reproducibility
- Enforce structural clarity
- Separate exploration from formal experiments
- Ensure traceability of results
- Make code understandable by humans and agents
- Prevent entropy over time

These standards apply to all projects unless explicitly overridden by a project-level `PROJECT_OVERRIDES.md`.

---

# 1. Core Philosophy

1. Exploration and experiments must be separated.
2. No important result should be irreproducible.
3. All experiments must be config-driven.
4. Code in `src/` must be reusable and modular.
5. No hidden state or implicit configuration.
6. Every artifact must be traceable to a config and commit.

---

# 2. Directory Conventions

Each project should follow this structure unless explicitly justified:

```
project_root/
│
├── README.md
├── PROJECT_STANDARDS.md (optional local override)
│
├── configs/
├── src/
├── experiments/
├── exploration/
├── tests/
└── outputs/
```

## Directory Roles

### src/
Reusable code only.
- No ad-hoc experiment scripts
- No notebooks
- No hardcoded parameters
- Must be modular and importable

### experiments/
Reproducible experiments.
- Each experiment gets its own folder
- Must include config
- Must save outputs
- Must log metadata

### exploration/
Sandbox only.
- Jupyter notebooks allowed
- Quick scripts allowed
- Non-reproducible work allowed
- Cannot be cited as final results

### configs/
All experiment parameters must live here.
No hardcoded experimental parameters in code.

### outputs/
Generated artifacts only.
Never commit large binary artifacts unless required.

---

# 3. Naming Conventions

## Directories

Experiments must follow:

YYYY_MM_DD_short_slug/

Example:

2026_02_22_activation_patching/
2026_03_01_sparse_autoencoder/

## Python Files

Modules:
snake_case.py

Classes:
PascalCase

Functions:
snake_case()

Constants:
UPPER_CASE

Private functions:
_leading_underscore()

---

# 4. Experiment Standards

Every experiment must:

1. Be config-driven
2. Set and log random seed
3. Log git commit hash
4. Save a copy of the config used
5. Save outputs under a structured directory
6. Produce deterministic results when possible

Each experiment folder should contain:

run.py  
config.yaml  
notes.md  
results/

## run.py must:

- Load config
- Set seed
- Execute experiment
- Save outputs
- Save metadata.json

---

# 5. Configuration Standards

Rules:

- No hardcoded hyperparameters
- No hardcoded model names
- No hidden default behavior

All experiment parameters must come from a config file.

Config files must be YAML or JSON.

---

# 6. Logging and Metadata

Each experiment must log:

- Git commit hash
- Timestamp
- Random seed
- Full config
- Environment information (if relevant)

Saved as:

results/metadata.json

---

# 7. Reproducibility Tiers

Every experiment should be classified:

- Tier 0 — Exploration only
- Tier 1 — Script reproducible
- Tier 2 — Config reproducible
- Tier 3 — Deterministic + logged
- Tier 4 — Publication-ready

Tier level must be noted in notes.md.

---

# 8. Code Quality Standards

- Functions must be small and single-purpose.
- Avoid global state.
- Avoid hidden mutation.
- Avoid circular dependencies.
- Write type hints when possible.
- Prefer explicit over implicit logic.

If a function exceeds ~60 lines, consider refactoring.

---

# 9. Testing Standards

Minimum requirements:

- Core logic must have unit tests.
- Shape assumptions must be tested.
- Hook or callback logic must be tested (if applicable).
- Test files go in tests/.

No experiment should rely solely on visual inspection.

---

# 10. Exploration Rules

The following are allowed in exploration/:

- Notebooks
- Temporary scripts
- Interactive probing
- Quick diagnostics

The following are forbidden in exploration/:

- Final figures for publication
- Long-term maintained abstractions
- Reusable core logic

Reusable logic must be migrated to src/.

---

# 11. Forbidden Patterns

The following are prohibited:

- Hardcoding experiment parameters
- Writing directly into project root
- Saving outputs without metadata
- Using global mutable state for configuration
- Mixing exploration and experiment logic

---

# 12. Agent Compliance

Any coding agent operating in this repository must:

1. Follow naming conventions.
2. Never hardcode experimental parameters.
3. Place reusable logic in src/.
4. Place exploratory work in exploration/.
5. Place reproducible work in experiments/.
6. Log metadata for experiments.

Non-compliant code should be considered invalid.

---

# 13. Change Policy

Standards may evolve.

Changes must:

- Preserve reproducibility
- Improve clarity
- Reduce entropy
- Increase structural coherence

---

# End of Standards
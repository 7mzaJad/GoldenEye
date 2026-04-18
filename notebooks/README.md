# Notebooks Folder

This folder is for exploratory analysis and experiments.

## Guidelines

- Keep notebooks focused and task-specific.
- Clear heavy outputs before committing when possible.
- Move reusable logic into `src/` modules instead of duplicating large code blocks in notebooks.
- Do not hardcode machine-specific absolute paths; prefer paths relative to repository root.

## Typical Use Cases

- Dataset exploration
- Quick metric checks
- Visualization of predictions

## Reproducibility Tip

When a notebook workflow becomes stable, create or update an equivalent script in `src/` and document it in the relevant README.

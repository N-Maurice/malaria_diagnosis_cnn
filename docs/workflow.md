# Team workflow

```text
main
  ├── maurice/*
  ├── laura/*
  ├── sarah/*
  └── david/*
```

1. `main` is protected by team policy. Enable GitHub branch protection after the
   initial push; local Git initialization cannot enforce remote protection.
2. No direct work is committed to `main` after initial setup.
3. Each member works primarily in their assigned member and notebook directories.
4. Shared utilities belong under `src/`.
5. Discuss shared utility changes with the team.
6. Give every meaningful change a descriptive commit.
7. Pull requests explain what changed, why, and how it was verified.
8. Never commit datasets.
9. Never commit model weights.
10. Never commit generated TensorBoard logs; use approved external artifact storage.
11. Do not overwrite another member's work.
12. Before merging, run `python -m unittest discover -s tests -v` and verify shared utilities.
13. Each member owns a separate notebook.
14. Do not silently modify another member's model.

Start from updated main, create a branch using the naming conventions, implement
inside your workspace, update your contribution/experiment logs, review
`git diff --cached`, push your branch, and open a reviewed pull request.
The maintainer should require review and passing checks through GitHub settings.

Agree on the dataset and approved split manifests once. Use the same manifests
for all models, seed 42, and validation for selection. Before each future run,
record its Git commit, manifest version, configuration, seed, and package versions
(`python -m pip freeze > members/<workspace>/results/environment.txt`). Exact
reproduction also depends on hardware, dependency versions and deterministic ops.
Use `set_random_seed()` before creating models and pipelines.

Keep generated artifacts in the ignored results/figures/gradcam and logs folders.
Share final artifacts separately; adding selected small figures to Git must be
an explicit team decision. Keep notebooks free of bulky embedded output.

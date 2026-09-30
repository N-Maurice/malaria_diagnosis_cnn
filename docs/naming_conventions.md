# Naming conventions

Branches: `<member>/<type>-<short-description>`, lowercase and short.
Allowed types: `feature`, `fix`, `docs`, `refactor`, `experiment`.
Examples: `yourname/feature-data-pipeline`, `maurice/feature-resnet`,
`laura/feature-transfer-model`, `sarah/feature-evaluation`, `david/fix-gradcam`.

Commits: `<type>: <short description>`.
Examples: `feat: add data split utility`, `feat: add evaluation metrics`,
`fix: correct dataset path handling`, `docs: update experiment workflow`,
`refactor: simplify logging utility`. Initial infrastructure may use `chore:`.
Avoid `update`, `changes`, `stuff`, `final`, and `done`.

Runs: `<model>_exp_<NN>_<description>`; keep experiment numbers unique per model.
Examples: `custom_resnet_exp_01_baseline`, `custom_resnet_exp_02_lr_1e-3`,
`mobilenet_exp_04_unfreeze_last20`, `efficientnet_exp_06_lr_1e-5`.
Avoid `test1`, `final`, `final_final`, `run1`, `experiment`.
Transfer architecture names are examples, not choices already made by the team.
Use snake_case for Python modules/functions, descriptive notebook filenames,
and the run name in artifact filenames. Record substantive changes in the log.

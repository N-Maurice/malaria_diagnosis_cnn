# Malaria Cell Image Classification

Reusable infrastructure for a four-member university project classifying NIH/NLM
cell images as **Uninfected (0)** or **Parasitized (1)**. This scaffold contains
no implemented architectures, training runs, experiment results, or report narrative.

## Team

| Member | Model | Ownership |
| --- | --- | --- |
| david | Custom ResNet | Model + experiments |
| sarah | Transfer Learning Model 1 | Model + experiments |
| laura | Transfer Learning Model 2 | Model + experiments |
| maurice | Additional Custom CNN | Model + experiments |

Each owner also produces their notebook, analysis and corresponding report contribution.

## Environment setup

Use a Python version supported by your chosen TensorFlow release; see the
[official installation guide](https://www.tensorflow.org/install/pip).
Requirements specify compatible lower bounds rather than a platform-specific lock.
Record installed versions for each future experiment.

```bash
git clone <repository-url> malaria-cnn
cd malaria-cnn
python -m venv .venv
```

For this existing workspace, enter `malaria_diagnosis_cnn` instead. Git clone's
optional destination above only changes the local directory name.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name malaria-cnn --display-name "Python (Malaria CNN)"
python -m unittest discover -s tests -v
```

The environment is local and ignored. Install dependencies on each clone. Tests
use synthetic fixtures only; TensorFlow-dependent checks explicitly skip if it is
absent. A complete environment should run them without skips.

## Dataset

The default dataset assumption is:

```text
data/cell_images/
├── Parasitized/
└── Uninfected/
```

This folder layout was observed locally. See [dataset notes](docs/dataset_notes.md)
for counts and assumptions still requiring verification. Place your independently
obtained dataset here; nothing is downloaded automatically or committed to Git.
Empty `data/Parasitized` and `data/Uninfected` placeholders also exist to match the
requested scaffold but are not searched by default. Never split both layouts.
Image discovery filters supported extensions beneath the two class directories.

After verifying the layout and leakage risks, one team member may explicitly run:

```bash
python -m scripts.create_splits
```

Commit the approved small CSV manifests and reuse them for every model. The
split is stratified 70/15/15 with seed 42. **Keep test data untouched during all
training and hyperparameter/threshold selection.** Existing manifests are protected
against overwrite. The scaffold itself has not generated any real-data splits.

## Project structure

- `src/`: shared configuration, discovery/splits/pipelines, metrics, TensorBoard and Grad-CAM.
- `data/`: ignored local dataset; only empty placeholders are tracked.
- `splits/`: shared portable `train_files.csv`, `val_files.csv`, `test_files.csv` when approved.
- `members/<owner>_<model>/`: model stub, blank experiment/error templates and ignored artifacts.
- `notebooks/<owner>/`: individual notebook workflow; every member authors their own notebook.
- `logs/fit/`: ignored TensorBoard run directories.
- `report/`: empty draft/figure folders and a bibliography placeholder.
- `docs/`: contribution tracking, naming, workflow, dataset notes and submission checklist.
- `scripts/`: explicit split-generation command; no automatic downloads.
- `tests/`: synthetic infrastructure verification, with no training or team architectures.
- `.venv/`: local Python environment, never committed.

## Running shared utilities

Run commands from the project root. These imports do not train or generate results:

```python
from src.config import DATASET_DIR, set_random_seed
from src.data_utils import build_dataframe, stratified_split, build_tf_dataset
from src.eval_utils import evaluate_binary, plot_confusion_matrix, plot_roc_curve
from src.logging_utils import make_run_name, create_tensorboard_callback, log_final_metrics
from src.gradcam_utils import make_gradcam_heatmap, overlay_heatmap

run_name = make_run_name("custom_resnet", 1, "baseline")
```

Use `build_dataframe(DATASET_DIR)` for image discovery and
`stratified_split(frame)` to create the agreed manifests once. Read a manifest
using `pandas.read_csv`; `build_tf_dataset(frame)` resolves paths from project root.
It defaults to 128×128 RGB, batch size 32, no shuffle/augmentation, pixels in [0,1].
Supply a callable for generic augmentation only on training. A supplied
`preprocessing_fn` receives resized float pixels in [0,255] and **replaces**
default normalization; use each future transfer model's documented contract.
Call `set_random_seed()` before constructing pipelines/models.

`evaluate_binary(labels, probabilities)` expects positive-class probabilities,
uses threshold 0.5 by default and returns all seven required metrics. Undefined
single-class rates/AUC are NaN. Plotting helpers return figures without displaying
them; pass `output_path` to save and close figures when finished.

`create_tensorboard_callback(model, number, description, hparams)` returns a
callback and unique run directory, refusing reuse. Required HParams keys:
`model_name`, `learning_rate`, `optimizer`, `batch_size`, `epochs`, `augmentation`,
`dropout`, `l2`, `frozen_layers`. Add batch normalization, architecture details,
Git commit, manifest identity and environment snapshot as extra fields.
The callback captures training/validation scalars; `log_final_metrics` writes
supplied scalar metrics under `validation/` or, after final selection, `test/`.
Undefined metrics must be omitted. HParams infrastructure follows the
[TensorBoard HParams API](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams).

When actual runs exist, view them with:

```bash
tensorboard --logdir logs/fit
```

Grad-CAM accepts a connected convolutional layer name or a callable returning
`(features, predictions)` from one connected forward pass. Subclassed models must
expose this access path. Inputs must have the model's preprocessing already
applied. Overlay uses the original uint8 RGB image. See module docstrings and the
[Keras Grad-CAM example](https://keras.io/examples/vision/grad_cam/) for the method.

## Team workflow and experiment naming

**main is protected.** This is team policy; enable GitHub branch protection after
the first push, because local Git initialization does not configure GitHub rules.
Work on `<member>/<type>-<short-description>` branches, commit with
`<type>: <short description>`, and open reviewed pull requests explaining changes,
reasons and verification. Discuss shared `src/` changes first. Do not overwrite
another member's work. See [workflow](docs/workflow.md) and
[naming conventions](docs/naming_conventions.md).

Runs use `<model>_exp_<NN>_<description>`, for example
`custom_resnet_exp_01_baseline`, `mobilenet_exp_04_unfreeze_last20`, or
`efficientnet_exp_06_lr_1e-5`. Transfer names are examples, not assigned choices.
Avoid `test1`, `final`, `final_final`, `run1`, and `experiment`.
Each owner must complete at least seven meaningful experiments and maintain their
blank templates. Preserve traceable configurations and TensorBoard evidence.

## GitHub connection

Create an empty GitHub repository first, then run from this project directory:

```bash
git remote add origin https://github.com/<GITHUB_USERNAME>/<REPOSITORY_NAME>.git
git branch -M main
git push -u origin main
```

After the initial push, configure `main` protection and pull-request review rules.
No GitHub repository or remote is created by this scaffold.

## Academic license

`LICENSE` is a project-specific educational-use template for this ALU student
project, explicitly unofficial. It does not claim institutional ownership or
cover third-party datasets, dependencies, or the existing assignment PDF.

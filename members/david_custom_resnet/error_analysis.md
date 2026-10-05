# Error analysis — David

Final model: `custom_resnet_exp_08_plateau_schedule`, evaluated once on the held-out test set
(4,134 images: 2,067 Parasitized, 2,067 Uninfected), threshold 0.5 on P(Parasitized).
Source: `runs/custom_resnet_exp_08_plateau_schedule/test_predictions.csv` and notebook Sections 13–15.
Figures: `MyDrive/malaria_cnn/david_custom_resnet/figures/` (`gradcam_incorrect.png`,
`errors_false_negatives.png`, `errors_false_positives.png`, `probability_histogram.png`).

## 1. Misclassified examples

136 of 4,134 test cells (3.29%) are misclassified:

| Error type | Count | Rate |
|---|---|---|
| False negatives (Parasitized → Uninfected, missed infection) | 89 | 4.31% of infected cells |
| False positives (Uninfected → Parasitized, false alarm) | 47 | 2.27% of uninfected cells |

Confidence of the mistakes: 52 of 136 are borderline (0.3 < p < 0.7). 28 are highly confident
(p < 0.05 or p > 0.95).

Most confident mistakes (shown with Grad-CAM in `gradcam_incorrect.png`):

| # | Image | True | P(Parasitized) | Predicted | Visual observation |
|---|---|---|---|---|---|
| 1 | `Uninfected/C75P36_ThinF_IMG_20150815_163707_cell_33.png` | Uninfected | 0.9989 | Parasitized | distinct purple-stained inclusion near the upper-left edge; Grad-CAM focuses on it |
| 2 | `Uninfected/C128P89ThinF_IMG_20151004_131753_cell_99.png` | Uninfected | 0.9983 | Parasitized | purple-stained structure on the right side of the cell; Grad-CAM focuses on it |
| 3 | `Parasitized/C100P61ThinF_IMG_20150918_144348_cell_142.png` | Parasitized | 0.0035 | Uninfected | no visible purple inclusion; Grad-CAM diffuse over the cell body |
| 4 | `Parasitized/C179P140ThinF_IMG_20151127_153420_cell_174.png` | Parasitized | 0.0038 | Uninfected | pale grey-blue cell (different colour from typical pink cells), no visible inclusion; Grad-CAM diffuse |
| 5 | `Parasitized/C122P83ThinF_IMG_20151002_145014_cell_171.png` | Parasitized | 0.0039 | Uninfected | only a very small dark brownish dot at the bottom edge of the cell; rest of cell clear |
| 6 | `Uninfected/C69P30N_ThinF_IMG_20150819_135421_cell_157.png` | Uninfected | 0.9939 | Parasitized | clear magenta-stained inclusion in the centre of the cell, visually similar to infected examples |

The full list is in `figures/misclassified_test_cells.csv`.

## 2. True label

See the table above (folder name = ground-truth label in the NIH dataset).

## 3. Predicted label

See the table above (threshold 0.5).

## 4. Predicted probability (Parasitized = 1) and threshold

Threshold 0.5, not tuned. Validation predictions for every run are in `runs/<run>/val_predictions.csv`.
Any threshold change must be chosen on those, never on the test set.

## 5. Possible cause

_David to write._ Points to weigh (verify each against the images before claiming it):

- Examples 1, 2 and 6: a visible stained inclusion in a cell labelled Uninfected. Candidates: label noise,
  a platelet or stain debris overlapping the cell, or an artefact that looks like a parasite.
- Examples 3–5: no clearly visible parasite (example 5 only a tiny dark dot at the edge) in a cell labelled Parasitized. Candidates: label noise, a parasite
  outside the cropped region or out of focus, or a staining/illumination difference (example 4).
- The borderline errors (52 with 0.3 < p < 0.7) are a different group from the confident ones and may
  have a different cause.

## 6. Evidence (image path, run name, split and heatmap)

Run `custom_resnet_exp_08_plateau_schedule`, test split, paths above, heatmaps in `gradcam_incorrect.png`.

## 7. Overfitting/underfitting observations

Train vs validation accuracy at the restored (best) epoch, from `experiments_summary.csv`:

| Run | Train acc | Val acc | Gap (train − val) | Best / completed epochs |
|---|---|---|---|---|
| exp_01_baseline | 95.89% | 96.27% | −0.38 | 6 / 11 |
| exp_02_deeper_4stage | 96.29% | 96.47% | −0.18 | 6 / 11 |
| exp_03_no_skip_ablation | 96.74% | 96.54% | +0.20 | 20 / 20 |
| exp_04_augmentation | 91.02% | 95.69% | −4.67 | 1 / 6 |
| exp_05_dropout_l2 | 95.29% | 95.57% | −0.28 | 7 / 12 |
| exp_06_lr_3e-4 | 96.39% | 96.54% | −0.15 | 20 / 20 |
| exp_07_sgd_momentum | 96.72% | 96.76% | −0.04 | 20 / 20 |
| exp_08_plateau_schedule | 97.35% | 97.24% | +0.11 | 30 / 30 |

Curve evidence (validation comparison plot `experiment_comparison.png` and final curves
`final_accuracy_loss_curves.png`):

- No run shows a large train > validation gap at its best epoch. Gaps are under 0.4 points except Exp 04.
  In Exp 04, training accuracy is below validation because augmentation is applied to training images only,
  and the restored epoch is epoch 1.
- Validation accuracy and loss have isolated spikes: Exp 01 at epoch 4 (val acc ≈ 0.75); Exp 07 at
  epoch 10 (val acc ≈ 0.73, val loss ≈ 1.3); Exp 08 at epochs 10–11 and 16. Training curves stay smooth
  at those epochs.
- Exp 01 and 02 stopped at epoch 11 with best epoch 6. Exp 03, 06, 07 and 08 ended at their epoch cap
  with the best epoch being the last one.
- Final model curves: training and validation accuracy/loss track closely; the last epochs show
  train 97.35% / val 97.24% accuracy.
- Loss values for Exp 05–08 include the L2 penalty (Keras adds it to reported loss), so their loss
  curves are only comparable with each other.

_David to write: which symptom(s) these show (overfitting, underfitting, optimisation instability), citing
the runs above._

## 8. What was attempted

Changes made across runs (see `experiments_log.md`):

- More capacity (Exp 02).
- Augmentation (Exp 04).
- Dropout + L2 (Exp 05).
- Lower learning rate (Exp 06).
- SGD + momentum (Exp 07).
- ReduceLROnPlateau with longer training (Exp 08).

## 9. Result of the attempted fix

| Change | Val accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|
| Before (Exp 02) | 96.47% | 95.60% | 97.34% | 0.9932 |
| + augmentation (Exp 04) | 95.69% | 94.29% | 97.10% | 0.9892 |
| + dropout/L2 (Exp 05) | 95.57% | 93.13% | 98.02% | 0.9888 |
| + LR 3e-4 (Exp 06) | 96.54% | 95.36% | 97.73% | 0.9935 |
| + plateau schedule, 30 epochs (Exp 08) | 97.24% | 96.42% | 98.06% | 0.9956 |

_David to interpret._

## 10. Remaining limitations

- **Patient overlap between splits.** The shared split is random at the image level. File names carry a
  slide/patient code (e.g. `C75P36`). All 148 patient codes found in test file names also occur in the
  training split (555 test files have a different naming pattern and were not parsed). The test score
  therefore measures generalisation to new cells from known patients, not to unseen patients. This
  applies to all four group models.
- Exp 08's best epoch was its last (30/30), so it may not have fully converged.
- Single run per configuration (seed 42); run-to-run variance was not measured.
- Threshold fixed at 0.5; no sensitivity-oriented threshold was chosen on validation data.
- Ground-truth label noise cannot be ruled out (examples 1–6).

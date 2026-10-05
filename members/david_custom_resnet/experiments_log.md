# Custom ResNet — David

Experiment record for the Custom ResNet (`members/david_custom_resnet/`). All runs were trained on
Google Colab (T4 GPU) with notebook `notebooks/david/custom_resnet_malaria.ipynb`.

**Common to every run:** code commit `eced63e`; shared split manifests (SHA-256 prefix `c5f90e8c1a71`,
70/15/15 stratified, seed 42); 128×128 RGB input scaled to [0, 1]; batch size 32; binary cross-entropy;
early stopping on validation loss with best weights restored; BatchNorm on; trained from scratch
(no pretrained weights, no frozen layers); decision threshold 0.5. Model selection used the validation
set only. The test set was evaluated once, for the selected final run.

TensorBoard: `MyDrive/malaria_cnn/david_custom_resnet/logs/fit/<run name>` (HParams + train/validation
scalars per epoch + `validation/*` final metrics; `test/*` for the final run only).
Per-run artefacts: `MyDrive/malaria_cnn/david_custom_resnet/runs/<run name>/`.

## Summary (validation set)

Sensitivity = recall on Parasitized; specificity = true-negative rate on Uninfected.

| Run | Changed vs base | Acc | Precision | Sensitivity | Specificity | F1 | ROC-AUC | Best / run epochs | Train / val acc at best | Min |
|---|---|---|---|---|---|---|---|---|---|---|
| `custom_resnet_exp_01_baseline` | — (reference) | 96.27% | 97.85% | 94.63% | 97.92% | 96.21% | 0.9925 | 6 / 11 | 95.89% / 96.27% | 6.1 |
| `custom_resnet_exp_02_deeper_4stage` | depth: 3→4 stages, 1→2 blocks/stage | 96.47% | 97.29% | 95.60% | 97.34% | 96.44% | 0.9932 | 6 / 11 | 96.29% / 96.47% | 7.6 |
| `custom_resnet_exp_03_no_skip_ablation` | skip connections removed (ablation, not carried forward) | 96.54% | 97.06% | 95.98% | 97.10% | 96.52% | 0.9934 | 20 / 20 | 96.74% / 96.54% | 12.7 |
| `custom_resnet_exp_04_augmentation` | data augmentation added | 95.69% | 97.01% | 94.29% | 97.10% | 95.63% | 0.9892 | 1 / 6 | 91.02% / 95.69% | 4.4 |
| `custom_resnet_exp_05_dropout_l2` | dropout 0.3 + L2 1e-4 | 95.57% | 97.91% | 93.13% | 98.02% | 95.46% | 0.9888 | 7 / 12 | 95.29% / 95.57% | 8.3 |
| `custom_resnet_exp_06_lr_3e-4` | learning rate 1e-3 → 3e-4 | 96.54% | 97.67% | 95.36% | 97.73% | 96.50% | 0.9935 | 20 / 20 | 96.39% / 96.54% | 14.0 |
| `custom_resnet_exp_07_sgd_momentum` | optimizer Adam → SGD+Nesterov, LR 1e-2 | 96.76% | 96.49% | 97.05% | 96.47% | 96.77% | 0.9936 | 20 / 20 | 96.72% / 96.76% | 13.8 |
| `custom_resnet_exp_08_plateau_schedule` | LR schedule + 30 epochs (early-stopping patience 7) | 97.24% | 98.03% | 96.42% | 98.06% | 97.22% | 0.9956 | 30 / 30 | 97.35% / 97.24% | 20.6 |

**Final selected run:** `custom_resnet_exp_08_plateau_schedule`.

## Final model — held-out test set (4,134 images, evaluated once)

| Accuracy | Precision | Sensitivity (recall) | Specificity | F1 | ROC-AUC | False negatives | False positives |
|---|---|---|---|---|---|---|---|
| 96.71% | 97.68% | 95.69% | 97.73% | 96.68% | 0.9946 | 89 / 2,067 | 47 / 2,067 |

Target (custom model): sensitivity ≥ 90% and specificity ≥ 90% — met.

## Notes for reading the logs

- **Loss values are not comparable across all runs.** Keras adds the L2 penalty to the reported training
  and validation loss, so Exp 05–08 (L2 = 1e-4) report higher losses than Exp 01–04 for the same quality of
  fit. Exp 07's ~0.41 loss mostly reflects the L2 term under SGD, not poor classification. Compare runs on
  the validation metrics above, and compare loss curves only between runs with the same L2 setting.
- **Runs that hit their epoch cap** (best epoch = last epoch): Exp 03, 06, 07 (20/20) and Exp 08 (30/30).
  These may not have fully converged.
- **Run time is total, not per epoch.** Per-epoch time: Exp 02 ≈ 0.69 min, Exp 03 ≈ 0.64 min.
- **Validation set size:** 4,134 images, so 1 percentage point ≈ 41 images. Differences of a few tenths
  of a point between runs are within run-to-run noise.

## Experiment 01

| Field | Entry |
| --- | --- |
| Run name | `custom_resnet_exp_01_baseline` |
| Date | October 2026 (Colab, T4 GPU) |
| Hypothesis | A small 3-stage ResNet with BatchNorm establishes the reference performance. |
| Configuration / Git commit / split identity / seed | derived from —; commit `eced63e`; splits `c5f90e8c1a71`; seed 42 |
| Learning rate | 1e-3 (constant) |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs (planned / completed) | 20 / 11 (best epoch 6; early-stopping patience 5) |
| Augmentation | none |
| Dropout | 0.0 |
| L2 | 0 |
| Batch normalization | yes |
| Skip connections | yes |
| Frozen/unfrozen layers | n/a — trained from scratch |
| Architecture | 3 stages 32-64-128, 1 block/stage; 308,833 parameters |
| Training result (at best epoch) | accuracy 95.89%, loss 0.1228 |
| Validation result | accuracy 96.27%, precision 97.85%, recall/sensitivity 94.63%, specificity 97.92%, F1 96.21%, ROC-AUC 0.9925; val loss 0.1119 |
| Test result (final selected model only) | — (not evaluated on test) |
| TensorBoard path / evidence | `logs/fit/custom_resnet_exp_01_baseline` |
| Conclusion | _David to write_ |
| Next decision | _David to write_ |

## Experiment 02

| Field | Entry |
| --- | --- |
| Run name | `custom_resnet_exp_02_deeper_4stage` |
| Date | October 2026 (Colab, T4 GPU) |
| Hypothesis | More residual blocks and a 4th stage add capacity and a larger receptive field. |
| Configuration / Git commit / split identity / seed | derived from Exp 01; commit `eced63e`; splits `c5f90e8c1a71`; seed 42 |
| Learning rate | 1e-3 (constant) |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs (planned / completed) | 20 / 11 (best epoch 6; early-stopping patience 5) |
| Augmentation | none |
| Dropout | 0.0 |
| L2 | 0 |
| Batch normalization | yes |
| Skip connections | yes |
| Frozen/unfrozen layers | n/a — trained from scratch |
| Architecture | 4 stages 32-64-128-256, 2 blocks/stage; 2,800,097 parameters |
| Training result (at best epoch) | accuracy 96.29%, loss 0.1103 |
| Validation result | accuracy 96.47%, precision 97.29%, recall/sensitivity 95.60%, specificity 97.34%, F1 96.44%, ROC-AUC 0.9932; val loss 0.1039 |
| Test result (final selected model only) | — (not evaluated on test) |
| TensorBoard path / evidence | `logs/fit/custom_resnet_exp_02_deeper_4stage` |
| Conclusion | _David to write_ |
| Next decision | _David to write_ |

## Experiment 03

| Field | Entry |
| --- | --- |
| Run name | `custom_resnet_exp_03_no_skip_ablation` |
| Date | October 2026 (Colab, T4 GPU) |
| Hypothesis | Removing shortcuts from the deeper net slows optimisation / lowers accuracy, isolating the residual effect. |
| Configuration / Git commit / split identity / seed | derived from Exp 02; commit `eced63e`; splits `c5f90e8c1a71`; seed 42 |
| Learning rate | 1e-3 (constant) |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs (planned / completed) | 20 / 20 (best epoch 20; early-stopping patience 5) |
| Augmentation | none |
| Dropout | 0.0 |
| L2 | 0 |
| Batch normalization | yes |
| Skip connections | **no** (plain CNN, identical layers) |
| Frozen/unfrozen layers | n/a — trained from scratch |
| Architecture | 4 stages 32-64-128-256, 2 blocks/stage; 2,755,297 parameters |
| Training result (at best epoch) | accuracy 96.74%, loss 0.0929 |
| Validation result | accuracy 96.54%, precision 97.06%, recall/sensitivity 95.98%, specificity 97.10%, F1 96.52%, ROC-AUC 0.9934; val loss 0.0993 |
| Test result (final selected model only) | — (not evaluated on test) |
| TensorBoard path / evidence | `logs/fit/custom_resnet_exp_03_no_skip_ablation` |
| Conclusion | _David to write_ |
| Next decision | _David to write_ |

## Experiment 04

| Field | Entry |
| --- | --- |
| Run name | `custom_resnet_exp_04_augmentation` |
| Date | October 2026 (Colab, T4 GPU) |
| Hypothesis | Cells have no fixed orientation and staining varies; flips/rotations/colour jitter should reduce overfitting. |
| Configuration / Git commit / split identity / seed | derived from Exp 02; commit `eced63e`; splits `c5f90e8c1a71`; seed 42 |
| Learning rate | 1e-3 (constant) |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs (planned / completed) | 20 / 6 (best epoch 1; early-stopping patience 5) |
| Augmentation | flip_rot_color (h/v flips, 90° rotations, brightness ±20, contrast 0.85–1.15) |
| Dropout | 0.0 |
| L2 | 0 |
| Batch normalization | yes |
| Skip connections | yes |
| Frozen/unfrozen layers | n/a — trained from scratch |
| Architecture | 4 stages 32-64-128-256, 2 blocks/stage; 2,800,097 parameters |
| Training result (at best epoch) | accuracy 91.02%, loss 0.2317 |
| Validation result | accuracy 95.69%, precision 97.01%, recall/sensitivity 94.29%, specificity 97.10%, F1 95.63%, ROC-AUC 0.9892; val loss 0.1273 |
| Test result (final selected model only) | — (not evaluated on test) |
| TensorBoard path / evidence | `logs/fit/custom_resnet_exp_04_augmentation` |
| Conclusion | _David to write_ |
| Next decision | _David to write_ |

## Experiment 05

| Field | Entry |
| --- | --- |
| Run name | `custom_resnet_exp_05_dropout_l2` |
| Date | October 2026 (Colab, T4 GPU) |
| Hypothesis | Dropout before the classifier and L2 weight decay further narrow the train/validation gap. |
| Configuration / Git commit / split identity / seed | derived from Exp 04; commit `eced63e`; splits `c5f90e8c1a71`; seed 42 |
| Learning rate | 1e-3 (constant) |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs (planned / completed) | 20 / 12 (best epoch 7; early-stopping patience 5) |
| Augmentation | flip_rot_color |
| Dropout | 0.3 |
| L2 | 1e-4 |
| Batch normalization | yes |
| Skip connections | yes |
| Frozen/unfrozen layers | n/a — trained from scratch |
| Architecture | 4 stages 32-64-128-256, 2 blocks/stage; 2,800,097 parameters |
| Training result (at best epoch) | accuracy 95.29%, loss 0.1878 |
| Validation result | accuracy 95.57%, precision 97.91%, recall/sensitivity 93.13%, specificity 98.02%, F1 95.46%, ROC-AUC 0.9888; val loss 0.1784 |
| Test result (final selected model only) | — (not evaluated on test) |
| TensorBoard path / evidence | `logs/fit/custom_resnet_exp_05_dropout_l2` |
| Conclusion | _David to write_ |
| Next decision | _David to write_ |

## Experiment 06

| Field | Entry |
| --- | --- |
| Run name | `custom_resnet_exp_06_lr_3e-4` |
| Date | October 2026 (Colab, T4 GPU) |
| Hypothesis | A smaller step size gives smoother validation curves and a better minimum. |
| Configuration / Git commit / split identity / seed | derived from Exp 05; commit `eced63e`; splits `c5f90e8c1a71`; seed 42 |
| Learning rate | 3e-4 (constant) |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs (planned / completed) | 20 / 20 (best epoch 20; early-stopping patience 5) |
| Augmentation | flip_rot_color |
| Dropout | 0.3 |
| L2 | 1e-4 |
| Batch normalization | yes |
| Skip connections | yes |
| Frozen/unfrozen layers | n/a — trained from scratch |
| Architecture | 4 stages 32-64-128-256, 2 blocks/stage; 2,800,097 parameters |
| Training result (at best epoch) | accuracy 96.39%, loss 0.1617 |
| Validation result | accuracy 96.54%, precision 97.67%, recall/sensitivity 95.36%, specificity 97.73%, F1 96.50%, ROC-AUC 0.9935; val loss 0.1554 |
| Test result (final selected model only) | — (not evaluated on test) |
| TensorBoard path / evidence | `logs/fit/custom_resnet_exp_06_lr_3e-4` |
| Conclusion | _David to write_ |
| Next decision | _David to write_ |

## Experiment 07

| Field | Entry |
| --- | --- |
| Run name | `custom_resnet_exp_07_sgd_momentum` |
| Date | October 2026 (Colab, T4 GPU) |
| Hypothesis | SGD with Nesterov momentum (the optimiser used by He et al.) may generalise better than Adam. |
| Configuration / Git commit / split identity / seed | derived from Exp 06; commit `eced63e`; splits `c5f90e8c1a71`; seed 42 |
| Learning rate | 1e-2 (constant) |
| Optimizer | SGD (momentum 0.9, Nesterov) |
| Batch size | 32 |
| Epochs (planned / completed) | 20 / 20 (best epoch 20; early-stopping patience 5) |
| Augmentation | flip_rot_color |
| Dropout | 0.3 |
| L2 | 1e-4 |
| Batch normalization | yes |
| Skip connections | yes |
| Frozen/unfrozen layers | n/a — trained from scratch |
| Architecture | 4 stages 32-64-128-256, 2 blocks/stage; 2,800,097 parameters |
| Training result (at best epoch) | accuracy 96.72%, loss 0.403 |
| Validation result | accuracy 96.76%, precision 96.49%, recall/sensitivity 97.05%, specificity 96.47%, F1 96.77%, ROC-AUC 0.9936; val loss 0.4144 |
| Test result (final selected model only) | — (not evaluated on test) |
| TensorBoard path / evidence | `logs/fit/custom_resnet_exp_07_sgd_momentum` |
| Conclusion | _David to write_ |
| Next decision | _David to write_ |

## Experiment 08

| Field | Entry |
| --- | --- |
| Run name | `custom_resnet_exp_08_plateau_schedule` |
| Date | October 2026 (Colab, T4 GPU) |
| Hypothesis | Halving the LR when validation loss plateaus, with more epochs, lets the best config converge further. |
| Configuration / Git commit / split identity / seed | derived from Exp 06; commit `eced63e`; splits `c5f90e8c1a71`; seed 42 |
| Learning rate | 3e-4 (ReduceLROnPlateau (×0.5, patience 2, min 1e-6)) |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs (planned / completed) | 30 / 30 (best epoch 30; early-stopping patience 7) |
| Augmentation | flip_rot_color |
| Dropout | 0.3 |
| L2 | 1e-4 |
| Batch normalization | yes |
| Skip connections | yes |
| Frozen/unfrozen layers | n/a — trained from scratch |
| Architecture | 4 stages 32-64-128-256, 2 blocks/stage; 2,800,097 parameters |
| Training result (at best epoch) | accuracy 97.35%, loss 0.1278 |
| Validation result | accuracy 97.24%, precision 98.03%, recall/sensitivity 96.42%, specificity 98.06%, F1 97.22%, ROC-AUC 0.9956; val loss 0.1356 |
| Test result (final selected model only) | see *Final model — held-out test set* above |
| TensorBoard path / evidence | `logs/fit/custom_resnet_exp_08_plateau_schedule` |
| Conclusion | _David to write_ |
| Next decision | _David to write_ |

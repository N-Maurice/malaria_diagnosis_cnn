# Transfer Learning Model 1 — Sarah

EfficientNetB0, ImageNet pretrained. 7 experiments, each isolating one variable.
Validation used for selection; test set evaluated once, on the final selected model only.

Note on split: trained on an independently generated 70/15/15 stratified split
(seed 42), not the shared `splits/` manifest. Counts matched exactly
(19,290/4,134/4,134), but file-level identity with the shared manifest was not
verified. Flagging for team awareness.

## Experiment 01 — baseline

| Field | Entry |
| --- | --- |
| Run name | efficientnetb0_exp_01_baseline |
| Hypothesis | Frozen pretrained base as feature extractor sets a baseline |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs | 5 |
| Augmentation | flip, rotation, zoom |
| Dropout | 0.3 |
| Frozen/unfrozen layers | fully frozen base |
| Validation result | acc 0.9456, precision 0.9590, sensitivity 0.9318, AUC 0.9824 |
| Conclusion | Strong baseline but below sensitivity target |
| Next decision | Unfreeze top layers for fine-tuning |

## Experiment 02 — unfreeze last 20 layers

| Field | Entry |
| --- | --- |
| Run name | efficientnetb0_exp_02_unfreeze_last20 |
| Hypothesis | Fine-tuning top layers improves task-specific features |
| Learning rate | 1e-5 |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs | 5 |
| Dropout | 0.3 |
| Frozen/unfrozen layers | last 20 layers unfrozen |
| Validation result | acc 0.9545, precision 0.9633, sensitivity 0.9457, AUC 0.9893 |
| Conclusion | Clear improvement over frozen baseline |
| Next decision | Train longer with early stopping |

## Experiment 03 — same + 15 epochs, early stopping

| Field | Entry |
| --- | --- |
| Run name | efficientnetb0_exp_03_unfreeze_15epochs |
| Hypothesis | Model had not converged at 5 epochs |
| Learning rate | 1e-5 |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs | up to 15, early stopping on val_loss (patience 3) |
| Dropout | 0.3 |
| Frozen/unfrozen layers | last 20 layers unfrozen |
| Validation result | acc 0.9625, precision 0.9679, sensitivity 0.9573, AUC 0.9922 |
| Conclusion | Val loss kept improving for all 15 epochs, no overfitting; first configuration to clear the 95% sensitivity target |
| Next decision | Compare optimizer choice |

## Experiment 04 — SGD optimizer

| Field | Entry |
| --- | --- |
| Run name | efficientnetb0_exp_04_sgd_optimizer |
| Hypothesis | SGD with momentum may generalize differently than Adam |
| Learning rate | 1e-3 |
| Optimizer | SGD, momentum 0.9 |
| Batch size | 32 |
| Epochs | up to 15, early stopping |
| Dropout | 0.3 |
| Frozen/unfrozen layers | last 20 layers unfrozen |
| Validation result | acc 0.9625, precision 0.9679, sensitivity 0.9573, AUC 0.9921 |
| Conclusion | Nearly identical to Adam (exp 03); optimizer choice had little effect |
| Next decision | Test dropout rate |

## Experiment 05 — dropout 0.5

| Field | Entry |
| --- | --- |
| Run name | efficientnetb0_exp_05_dropout_05 |
| Hypothesis | Higher dropout may improve generalization |
| Learning rate | 1e-5 |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs | up to 15, early stopping |
| Dropout | 0.5 |
| Frozen/unfrozen layers | last 20 layers unfrozen |
| Validation result | acc 0.9625, precision 0.9689, sensitivity 0.9563, AUC 0.9918 |
| Conclusion | Negligible change from dropout 0.3, suggesting the model was not meaningfully overfitting at that configuration |
| Next decision | Test deeper fine-tuning |

## Experiment 06 — unfreeze last 40 layers (final selected model)

| Field | Entry |
| --- | --- |
| Run name | efficientnetb0_exp_06_unfreeze_last40 |
| Hypothesis | Fine-tuning more layers gives the model more capacity to adapt |
| Learning rate | 1e-5 |
| Optimizer | Adam |
| Batch size | 32 |
| Epochs | up to 15, early stopping |
| Dropout | 0.3 |
| Frozen/unfrozen layers | last 40 layers unfrozen |
| Validation result | acc 0.9642, precision 0.9667, sensitivity 0.9621, AUC 0.9934 |
| Test result (final model) | sensitivity 0.9536, specificity 0.9682, accuracy 0.96, AUC 0.9925 |
| Conclusion | Best-performing configuration; selected as final model |
| Next decision | Selected as final — proceed to evaluation, Grad-CAM, error analysis |

## Experiment 07 — batch size 64

| Field | Entry |
| --- | --- |
| Run name | efficientnetb0_exp_07_batch64 |
| Hypothesis | Larger batch size may change convergence behaviour |
| Learning rate | 1e-5 |
| Optimizer | Adam |
| Batch size | 64 |
| Epochs | up to 15, early stopping |
| Dropout | 0.3 |
| Frozen/unfrozen layers | last 40 layers unfrozen |
| Validation result | acc 0.9640, precision 0.9685, sensitivity 0.9597, AUC 0.9927 |
| Conclusion | Near-identical to experiment 06; fine-tuning depth was the dominant factor, not batch size |
| Next decision | None — experiment 06 remains the final model |

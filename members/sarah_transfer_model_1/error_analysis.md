# Error analysis — Sarah

## 1. Misclassified examples
162 of 4,134 test images (3.9% error rate) on the final selected model (experiment 06).

## 2. True label
65 misclassifications were actually Uninfected; 97 were actually Parasitized.

## 3. Predicted label
The 65 Uninfected cases were predicted Parasitized (false positives); the 97
Parasitized cases were predicted Uninfected (false negatives).

## 4. Predicted probability and threshold
Decision threshold: 0.5. False negatives examined had probabilities in the
0.10–0.46 range (low confidence, borderline). False positives examined had
probabilities in the 0.52–0.97 range (several confidently wrong).

## 5. Possible cause
False negatives showed faint, low-contrast staining compared to clearly infected
cells, consistent with early-stage or low-density infections being harder to
detect. False positives often contained irregular dark spots or debris that
visually resembled parasite markers without representing true infection.

## 6. Evidence
See figures/model6_error_examples.png (8 misclassified examples) and
gradcam/model6_gradcam.png (Grad-CAM on one correct infected, one correct
uninfected, and one incorrect case). Run: efficientnetb0_exp_06_unfreeze_last40.

## 7. Overfitting/underfitting observations
No overfitting observed. Training and validation loss curves stayed close and
both declined throughout training across all 7 experiments; validation metrics
tracked at or above training metrics due to dropout and augmentation being
active only during training.

## 8. What was attempted
Fine-tuning depth (20 vs. 40 unfrozen layers), extended training with early
stopping, dropout rate (0.3 vs. 0.5), optimizer choice (Adam vs. SGD), and
batch size (32 vs. 64) were all tested as regularization/capacity variables.

## 9. Result of the attempted fix
Fine-tuning depth had the largest effect — unfreezing the last 40 layers
(experiment 06) outperformed all other configurations. Dropout, optimizer, and
batch size changes had only marginal effect by comparison.

## 10. Remaining limitations
Errors concentrate in visually ambiguous cases (faint infections, staining
artefacts) rather than being randomly distributed. Evaluation used a single
train/test split rather than cross-validation, and the split was independently
generated rather than reused from the team's shared manifest (see note in
experiments_log.md).

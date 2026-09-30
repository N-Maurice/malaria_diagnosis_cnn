# David notebook workspace

Owner: David — Additional Custom CNN. Create your own notebook here; one notebook
per member is required. This is workflow guidance, not a shared implementation.
Start Jupyter from the project root and select **Python (Malaria CNN)**. If the
notebook kernel starts in this directory, use `%cd ../..` before importing `src`.
Call `set_random_seed()` before constructing pipelines or models.

1. Environment verification
2. Imports
3. Configuration
4. Dataset loading
5. Split loading
6. Data pipeline
7. Model definition
8. Model compilation
9. Experiment configuration
10. TensorBoard setup
11. Training
12. Evaluation
13. Visualization
14. Grad-CAM
15. Error analysis
16. Final experiment selection

Use validation during experiment selection. Evaluate the held-out test set only
after selection is frozen. Keep outputs small; store artifacts in your ignored
results/figures/gradcam folders. Do not embed large images or datasets in Git.

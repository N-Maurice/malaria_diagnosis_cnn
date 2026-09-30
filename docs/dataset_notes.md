# Dataset notes

## Confirmed requirements

- NIH/NLM malaria cell classification; expected 27,558 images.
- `Uninfected = 0`, `Parasitized = 1` (positive/infected class).
- Stratified image-level 70/15/15 train/validation/test, seed 42.
- Portable manifest paths relative to project root.
- Dataset contents are not committed, copied, or automatically downloaded.
- Test data remains untouched during training and hyperparameter experimentation.

## Local filesystem observations

Scaffold inspection found `data/cell_images/Parasitized` and
`data/cell_images/Uninfected`, each containing 13,779 PNG files plus `Thumbs.db`.
This confirms the local folder layout and extension counts only; image contents,
source authenticity, checksums, patient grouping and duplicate content were not
validated. No real images were decoded or used by scaffold tests.

## Assumptions requiring verification

The default loader assumes this layout on every member's computer:

```text
data/
└── cell_images/
    ├── Parasitized/*.png
    └── Uninfected/*.png
```

The initial requested tree also included `data/Parasitized` and `data/Uninfected`;
these are empty placeholders, not the default dataset location. Do not mix layouts.
For a deliberately flattened dataset pass `DATA_DIR` explicitly to discovery and
record that layout decision before generating shared manifests.

Images are assumed readable RGB-compatible cell images with folder-correct labels.
Discovery searches recursively beneath each class folder, sorts paths, and filters
PNG/JPG/JPEG/BMP suffixes case-insensitively. `Thumbs.db` is ignored. The pipeline
converts supported images to three-channel RGB. Decoding errors surface on use.
Verify exact extraction nesting, counts, corruption, provenance and usage terms
before running `python -m scripts.create_splits`. Obtain data independently through
the course-provided source; this repository performs no download.

The image-level split does not establish patient independence. Check patient IDs,
slide grouping and duplicate/near-duplicate images before adopting manifests;
if grouping is available, discuss a stratified group-aware split with the team
and instructor instead of claiming patient-level generalization. Filename
uniqueness checks do not detect identical image contents.

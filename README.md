# Chinese cabbage vision models

This repository contains the reproducibility materials for two tasks:

1. YOLO12 instance segmentation of non-heading Chinese cabbage.
2. Dual-view RGB-D fresh-weight estimation.

Large checkpoints and image/array datasets are distributed with the
[`v2.0` GitHub Release](https://github.com/18360151678/Dual-view-rgbd-fresh-weight-estimation/releases/tag/v2.0),
not in Git history. SHA-256 values are supplied with the release.

## Repository layout

```text
segmentation/
  evaluate.py                 YOLO12 test script
  data.yaml                   example dataset definition
fresh_weight/
  train.py                    three-seed training entry point
  evaluate.py                 validation/test entry point
  rgbd_experiment.py          model and statistical utilities
  data/                       split and validation-label manifests
  output/tensorboard/         selected training records
  results/                    Top-10 metrics and checkpoint manifest
scripts/
  prepare_release_assets.py   local release-packaging helper
```

## Data split for fresh-weight estimation

The split is performed at plant level so that the top and side views of one
plant always remain together. With split seed 42, the code first reserves 20%
of the 170 plants as an independent test set. It then divides the remaining
80% into training and validation subsets using an 80%/20% split. Integer
rounding gives 108 training plants, 28 validation plants, and 34 test plants.
The same manifest is reused for every architecture and training seed.

Training is repeated with seeds 42, 2024, and 3407. Batch size is 4 for every
backbone. The default optimizer settings are learning rate
`1e-4` and weight decay `1e-5`.

## Downloadable assets

The release contains:

- `yolo12_best.pt`;
- `segmentation_test_set.zip` with 204 images and 204 polygon-label files;
- ten fresh-weight checkpoints, one for each ranked model, selecting the seed
  with the highest validation R² within that model;
- `fresh_weight_validation_set.zip` with 34 paired top/side RGB-D samples and
  their fresh-weight labels;
- `checksums_sha256.txt`.

See the task-specific READMEs for extraction and evaluation instructions.


## License

Code is released under the MIT License. The released datasets are provided for
research use with attribution to the accompanying study.


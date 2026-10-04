# Dual-view RGB-D fresh-weight estimation

## Data layout

Each plant has one top-view and one side-view NumPy array. Each array must be
`H × W × 4`, with RGB in channels 0–2 and depth in channel 3.

```text
data/
  data.xlsx                   first column: sample ID; second: fresh weight (g)
  RGB_D/
    <id>_rgbd_top.npy
    <id>_rgbd_side.npy
```

## Train

`train.py` defaults to the published Top-10 configurations, batch size 4,
100 epochs, patience 20, pretrained backbone initialization, and seeds 42,
2024, and 3407.

```bash
pip install -r requirements.txt
python train.py
```

The script first reserves 20% of plants as an independent test set and then
uses 20% of the remaining plants for validation. It writes checkpoints,
TensorBoard events, histories, and `split_manifest.csv` below `output/`.

```bash
tensorboard --logdir output/runs
```

## Evaluate a new training run

Evaluate the untouched test split only after model selection is complete. A
reference model must be specified if paired test-set comparisons are requested.

```bash
python evaluate.py --split test --reference-model resnet18_shufflenet_mul
```

The report includes per-seed R², MAE, and RMSE, mean ± sample SD, bootstrap
confidence intervals, predictions, and paired bootstrap comparisons.

## Evaluate the released validation package

Download the ten `.pt` files and `fresh_weight_validation_set.zip` from the
release. Place the weights in `weights/` and extract the validation archive as
`validation_set/`, then run:

```bash
python evaluate.py \
  --weights-dir weights \
  --data-dir validation_set/RGB_D \
  --labels validation_set/validation_labels.csv \
  --all-data
```

This downloadable package contains the 34 plants from the original fixed
80% training and 20% validation experiment, so it matches the released
checkpoints and the archived Top-10 performance table. It is intentionally
separate from the new two-stage split implemented by `train.py`.

Only the highest-validation-R² seed checkpoint is released for each ranked
model. The corresponding TensorBoard event, epoch history, and selection
metadata are under `output/tensorboard/<model>/seed_<seed>/`. The three-seed
mean ± SD table remains available in `results/top10_metrics_mean_sd.csv`.


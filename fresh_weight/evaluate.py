from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from rgbd_experiment import (
    DEFAULT_SEEDS,
    DualBranchModel,
    MultiViewRGBDDataset,
    bootstrap_metrics,
    default_model_configs,
    holm_adjust,
    paired_bootstrap_difference,
    regression_metrics,
    split_indices,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "RGB_D")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "data.xlsx")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "output"
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Checkpoint directory. Default: output/runs from train.py.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Split manifest. Default: output/split_manifest.csv.",
    )
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Evaluate every row in --labels, for the released validation subset.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional training seeds to require for every model. When omitted, "
            "all discovered checkpoints are evaluated."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Evaluate the validation set or the untouched independent test set.",
    )
    parser.add_argument(
        "--expected-all-pairs",
        action="store_true",
        help="Require all 75 backbone/fusion configurations before validation reporting.",
    )
    parser.add_argument(
        "--reference-model",
        default=None,
        help=(
            "Pre-specified primary model for paired comparisons. On validation, "
            "the highest mean validation R2 is used when omitted. On test, paired "
            "comparisons are skipped when omitted."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def predict(model, loader, device: torch.device, label_scale: float) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
    for top, side, target, sample_ids in loader:
        output = model(
            top.to(device, non_blocking=True), side.to(device, non_blocking=True)
        )
        true_values = target.numpy().reshape(-1) * label_scale
        pred_values = output.cpu().numpy().reshape(-1) * label_scale
        if torch.is_tensor(sample_ids):
            normalized_ids = sample_ids.cpu().tolist()
        else:
            normalized_ids = list(sample_ids)
        rows.extend(
            {
                "sample_id": sample_id,
                "true": float(true),
                "pred": float(prediction),
            }
            for sample_id, true, prediction in zip(normalized_ids, true_values, pred_values)
        )
    return pd.DataFrame(rows).sort_values("sample_id", key=lambda col: col.astype(str))


def discover_checkpoints(weights_dir: Path, require_markers: bool) -> list[Path]:
    checkpoints = sorted(weights_dir.rglob("*.pt"))
    if require_markers:
        checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if (checkpoint.parent / "completed.json").exists()
        ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found below {weights_dir}")
    return checkpoints


def main() -> None:
    args = parse_args()
    weights_dir = args.weights_dir or (args.output_dir / "runs")
    checkpoints = discover_checkpoints(
        weights_dir, require_markers=args.weights_dir is None
    )
    selected_checkpoints: list[Path] = []
    checkpoint_keys: list[tuple[str, int]] = []
    for checkpoint_path in checkpoints:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        key = (str(saved["model_name"]), int(saved["training_seed"]))
        if args.seeds is None or key[1] in args.seeds:
            selected_checkpoints.append(checkpoint_path)
            checkpoint_keys.append(key)
    checkpoints = selected_checkpoints
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for requested seeds {args.seeds}.")
    seeds_by_model: dict[str, set[int]] = {}
    for model_name, seed in checkpoint_keys:
        seeds_by_model.setdefault(model_name, set()).add(seed)
    expected_seeds = set(args.seeds or [])
    incomplete = (
        {
            model_name: sorted(expected_seeds - seeds)
            for model_name, seeds in seeds_by_model.items()
            if seeds != expected_seeds
        }
        if args.seeds is not None
        else {}
    )
    duplicates = [key for key, count in Counter(checkpoint_keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate model/seed checkpoints found: {duplicates}")
    if incomplete:
        raise ValueError(f"Models missing requested training seeds: {incomplete}")
    if args.expected_all_pairs:
        expected_models = {config.name for config in default_model_configs(all_pairs=True)}
        observed_models = set(seeds_by_model)
        missing_models = sorted(expected_models - observed_models)
        if missing_models:
            raise ValueError(
                f"Validation reporting requires all 75 configurations; "
                f"{len(missing_models)} models are still missing. First missing models: "
                f"{missing_models[:10]}"
            )
    first_checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    dataset = MultiViewRGBDDataset(
        args.data_dir,
        args.labels,
        max_depth=float(first_checkpoint["max_depth"]),
        label_scale=float(first_checkpoint["label_scale"]),
    )
    if args.all_data:
        manifest = pd.DataFrame(
            {
                "row_index": np.arange(len(dataset)),
                "sample_id": dataset.ids,
                "target": dataset.weights,
                "split": "all",
            }
        )
        evaluation_idx = list(range(len(dataset)))
        evaluation_name = "all"
    else:
        manifest_path = args.manifest or (args.output_dir / "split_manifest.csv")
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run train.py first or use --all-data "
                "with a released subset."
            )
        manifest = pd.read_csv(manifest_path)
        manifest_ids = manifest["sample_id"].astype(str).tolist()
        if manifest_ids != list(map(str, dataset.ids)):
            raise ValueError("Split manifest IDs do not match the current label file.")
        evaluation_idx = split_indices(manifest, args.split)
        evaluation_name = args.split
    loader = DataLoader(
        Subset(dataset, evaluation_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for checkpoint_path in checkpoints:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = saved["model_config"]
        model = DualBranchModel(**config, pretrained=False).to(device)
        model.load_state_dict(saved["model_state"], strict=True)
        predictions = predict(model, loader, device, float(saved["label_scale"]))
        metrics = regression_metrics(predictions["true"], predictions["pred"])
        metric_rows.append(
            {
                "model": saved["model_name"],
                "seed": int(saved["training_seed"]),
                "best_epoch": int(saved["best_epoch"]),
                "best_val_r2": float(saved["best_val_r2"]),
                **metrics,
            }
        )
        predictions.insert(0, "seed", int(saved["training_seed"]))
        predictions.insert(0, "model", saved["model_name"])
        prediction_frames.append(predictions)
        print(f"Evaluated {saved['model_name']} seed={saved['training_seed']}: {metrics}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_seed = pd.DataFrame(metric_rows).sort_values(["model", "seed"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = (
        per_seed.groupby("model")[["r2", "mae", "rmse"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "model" if column[0] == "model" else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]

    ensemble_frames: dict[str, pd.DataFrame] = {}
    ci_rows: list[dict] = []
    for model_name, frame in predictions.groupby("model"):
        ensemble = (
            frame.groupby("sample_id", as_index=False)
            .agg(true=("true", "first"), pred=("pred", "mean"))
            .sort_values("sample_id", key=lambda col: col.astype(str))
        )
        ensemble_frames[model_name] = ensemble
        point = regression_metrics(ensemble["true"], ensemble["pred"])
        intervals, _ = bootstrap_metrics(
            ensemble["true"],
            ensemble["pred"],
            n_bootstrap=args.bootstrap,
            seed=args.bootstrap_seed,
        )
        for metric in ("r2", "mae", "rmse"):
            ci_rows.append(
                {
                    "model": model_name,
                    "prediction": "mean prediction across training seeds",
                    "metric": metric,
                    "point_estimate": point[metric],
                    "ci95_low": intervals[metric][0],
                    "ci95_high": intervals[metric][1],
                    "bootstrap_samples": args.bootstrap,
                }
            )

    if args.reference_model is not None:
        reference = args.reference_model
        if reference not in ensemble_frames:
            raise ValueError(f"Reference model '{reference}' has no predictions.")
    elif evaluation_name == "val":
        reference = per_seed.groupby("model")["best_val_r2"].mean().idxmax()
    else:
        reference = None

    comparison_rows: list[dict] = []
    if reference is not None:
        reference_frame = ensemble_frames[reference].set_index("sample_id")
        for candidate_name, candidate_frame_raw in ensemble_frames.items():
            if candidate_name == reference:
                continue
            candidate_frame = candidate_frame_raw.set_index("sample_id")
            joined = reference_frame[["true", "pred"]].join(
                candidate_frame[["pred"]], how="inner", rsuffix="_candidate"
            )
            rows = paired_bootstrap_difference(
                joined["true"],
                joined["pred_candidate"],
                joined["pred"],
                n_bootstrap=args.bootstrap,
                seed=args.bootstrap_seed,
            )
            for row in rows:
                row.update(
                    {"reference_model": reference, "candidate_model": candidate_name}
                )
                comparison_rows.append(row)

    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["p_value_holm"] = np.nan
        for metric, positions in comparisons.groupby("metric").groups.items():
            comparisons.loc[positions, "p_value_holm"] = holm_adjust(
                comparisons.loc[positions, "p_value_two_sided"]
            )
        comparisons["significant_after_holm_0.05"] = comparisons["p_value_holm"] < 0.05

    summary.insert(1, "n_training_seeds", per_seed.groupby("model").size().reindex(summary["model"]).values)
    report_path = args.output_dir / f"{evaluation_name}_results.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="mean_std", index=False)
        per_seed.to_excel(writer, sheet_name="per_seed_metrics", index=False)
        pd.DataFrame(ci_rows).to_excel(writer, sheet_name="bootstrap_ci", index=False)
        comparisons.to_excel(writer, sheet_name="paired_comparison", index=False)
        predictions.to_excel(writer, sheet_name="predictions", index=False)
        manifest.to_excel(writer, sheet_name="split_manifest", index=False)
        pd.DataFrame(
            [
                {
                    "reference_model": reference,
                    "selection_rule": (
                        "explicit --reference-model"
                        if args.reference_model
                        else (
                            "highest mean validation R2"
                            if evaluation_name == "val"
                            else "none; test-set model selection is prohibited"
                        )
                    ),
                    "evaluation_split": evaluation_name,
                    "evaluation_samples": len(evaluation_idx),
                    "bootstrap_samples": args.bootstrap,
                    "difference_definition": "candidate - reference",
                    "interpretation": "R2 > 0 favors candidate; MAE/RMSE < 0 favors candidate",
                }
            ]
        ).to_excel(writer, sheet_name="analysis_info", index=False)
    print(f"{evaluation_name.capitalize()} report written to {report_path}")
    print(f"Reference model for paired bootstrap comparisons: {reference or 'none'}")


if __name__ == "__main__":
    main()
